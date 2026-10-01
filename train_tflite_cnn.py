#!/usr/bin/env python3
"""Train the notebook-style casting CNN and export a full-INT8 TFLite model.

This pipeline deliberately uses no pretrained backbone. It trains a compact
grayscale convolutional network on the supplied casting dataset, evaluates a
held-out test split, and writes an Android-ready uint8 TensorFlow Lite model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import keras
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


CLASS_NAMES = ("def_front", "ok_front")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_SIZE = (300, 300)
IMAGE_CHANNELS = 1
DECISION_THRESHOLD = 0.5
MINIMUM_TFLITE_ACCURACY = 0.98
MAXIMUM_QUANTIZATION_ACCURACY_DROP = 0.01


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    label: int


@dataclass(frozen=True)
class DatasetSplits:
    train: tuple[ImageRecord, ...]
    validation: tuple[ImageRecord, ...]
    test: tuple[ImageRecord, ...]
    audit: dict[str, int]


@dataclass(frozen=True)
class ConversionResult:
    model_path: Path
    input_details: dict
    output_details: dict
    operators: tuple[str, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the notebook-style casting CNN and export full-INT8 TFLite."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/casting_data/casting_data"),
        help="Directory containing train/ and test/ class directories.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--representative-samples",
        type=int,
        default=200,
        help="Maximum balanced training images used for INT8 calibration.",
    )
    parser.add_argument(
        "--allow-low-accuracy",
        action="store_true",
        help="Keep artifacts and return success even if acceptance thresholds are missed.",
    )
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        default=None,
        help="Development-only cap applied per class and source split.",
    )
    args = parser.parse_args(argv)
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be between 0 and 0.5")
    for name in ("batch_size", "epochs", "representative_samples"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_images_per_split is not None and args.max_images_per_split <= 1:
        parser.error("--max-images-per-split must be greater than one")
    return args


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except RuntimeError:
        pass


def configure_tensorflow_memory_growth() -> None:
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def _image_files(class_dir: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png"}
    return sorted(path for path in class_dir.iterdir() if path.suffix.lower() in suffixes)


def _validate_image(path: Path) -> None:
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, UnidentifiedImageError) as error:
        raise ValueError(f"Unreadable image: {path}") from error


def _content_hash(path: Path) -> str:
    with path.open("rb") as image_file:
        return hashlib.file_digest(image_file, "sha256").hexdigest()


def _deduplicate_records(
    records: Sequence[ImageRecord], split_name: str
) -> tuple[tuple[ImageRecord, ...], int, dict[str, int]]:
    unique: list[ImageRecord] = []
    seen: dict[str, ImageRecord] = {}
    hashes: dict[str, int] = {}
    removed = 0
    for record in records:
        digest = _content_hash(record.path)
        hashes[digest] = record.label
        previous = seen.get(digest)
        if previous is None:
            seen[digest] = record
            unique.append(record)
            continue
        if previous.label != record.label:
            raise ValueError(
                f"Identical image has conflicting labels in {split_name}: "
                f"{previous.path} and {record.path}"
            )
        removed += 1
    return tuple(unique), removed, hashes


def _limit_per_class(
    records: Sequence[ImageRecord], limit: int | None
) -> tuple[ImageRecord, ...]:
    if limit is None:
        return tuple(records)
    limited: list[ImageRecord] = []
    for label in range(len(CLASS_NAMES)):
        class_records = [record for record in records if record.label == label]
        limited.extend(class_records[:limit])
    return tuple(limited)


def _stratified_split(
    records: Sequence[ImageRecord], validation_fraction: float, seed: int
) -> tuple[tuple[ImageRecord, ...], tuple[ImageRecord, ...]]:
    train: list[ImageRecord] = []
    validation: list[ImageRecord] = []
    for label in range(len(CLASS_NAMES)):
        class_records = [record for record in records if record.label == label]
        if len(class_records) < 2:
            raise ValueError(f"Class {CLASS_NAMES[label]} needs at least two training images")
        rng = np.random.default_rng(seed + label)
        indices = rng.permutation(len(class_records))
        validation_count = max(1, int(round(len(class_records) * validation_fraction)))
        validation_count = min(validation_count, len(class_records) - 1)
        validation_indices = set(indices[:validation_count].tolist())
        validation.extend(record for i, record in enumerate(class_records) if i in validation_indices)
        train.extend(record for i, record in enumerate(class_records) if i not in validation_indices)
    sort_key = lambda item: str(item.path)
    return tuple(sorted(train, key=sort_key)), tuple(sorted(validation, key=sort_key))


def discover_dataset(
    data_dir: Path,
    validation_fraction: float,
    seed: int,
    max_images_per_split: int | None = None,
) -> DatasetSplits:
    data_dir = data_dir.resolve()
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    for directory in (train_dir, test_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Required dataset directory does not exist: {directory}")

    def collect(split_dir: Path) -> tuple[ImageRecord, ...]:
        records: list[ImageRecord] = []
        for class_name, label in CLASS_TO_INDEX.items():
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Required class directory does not exist: {class_dir}")
            paths = _image_files(class_dir)
            if not paths:
                raise ValueError(f"No images found in {class_dir}")
            for path in paths:
                _validate_image(path)
                records.append(ImageRecord(path=path, label=label))
        return tuple(records)

    source_train = collect(train_dir)
    source_test = collect(test_dir)
    unique_test, removed_within_test, test_hashes = _deduplicate_records(source_test, "test")
    unique_train, removed_within_train, _ = _deduplicate_records(source_train, "train")

    filtered_train: list[ImageRecord] = []
    removed_train_test = 0
    for record in unique_train:
        digest = _content_hash(record.path)
        test_label = test_hashes.get(digest)
        if test_label is None:
            filtered_train.append(record)
        elif test_label != record.label:
            raise ValueError(f"Identical image has conflicting train/test labels: {record.path}")
        else:
            removed_train_test += 1

    all_train = _limit_per_class(filtered_train, max_images_per_split)
    test = _limit_per_class(unique_test, max_images_per_split)
    train, validation = _stratified_split(all_train, validation_fraction, seed)
    return DatasetSplits(
        train=train,
        validation=validation,
        test=test,
        audit={
            "source_train_images": len(source_train),
            "source_test_images": len(source_test),
            "removed_within_train_duplicates": removed_within_train,
            "removed_within_test_duplicates": removed_within_test,
            "removed_train_images_duplicated_in_test": removed_train_test,
        },
    )


def split_summary(splits: DatasetSplits) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for split_name in ("train", "validation", "test"):
        counts = {name: 0 for name in CLASS_NAMES}
        records = getattr(splits, split_name)
        for record in records:
            counts[CLASS_NAMES[record.label]] += 1
        counts["total"] = len(records)
        summary[split_name] = counts
    return summary


def decode_image(path: str | Path | tf.Tensor) -> tf.Tensor:
    if isinstance(path, Path):
        path = str(path)
    encoded = tf.io.read_file(path)
    image = tf.io.decode_image(encoded, channels=IMAGE_CHANNELS, expand_animations=False)
    image.set_shape((None, None, IMAGE_CHANNELS))
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear", antialias=True)
    return tf.clip_by_value(tf.cast(image, tf.float32), 0.0, 255.0)


def augment_image(image: tf.Tensor, index: tf.Tensor, seed: int) -> tf.Tensor:
    """Apply deterministic 0-10% zoom-in and 0-10% brightness reduction."""
    index = tf.cast(index % (2**31 - 1), tf.int32)
    crop_seed = tf.stack((tf.cast(seed, tf.int32), index))
    light_seed = tf.stack((tf.cast(seed + 1, tf.int32), index))
    zoom_factor = tf.random.stateless_uniform((), crop_seed, minval=0.9, maxval=1.0)
    crop_height = tf.cast(tf.round(zoom_factor * IMAGE_SIZE[0]), tf.int32)
    crop_width = tf.cast(tf.round(zoom_factor * IMAGE_SIZE[1]), tf.int32)
    image = tf.image.stateless_random_crop(
        image, (crop_height, crop_width, IMAGE_CHANNELS), seed=crop_seed
    )
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear", antialias=True)
    brightness = tf.random.stateless_uniform((), light_seed, minval=0.9, maxval=1.0)
    return tf.clip_by_value(image * brightness, 0.0, 255.0)


def make_dataset(
    records: Sequence[ImageRecord],
    batch_size: int,
    *,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    paths = [str(record.path) for record in records]
    labels = np.asarray([record.label for record in records], dtype=np.float32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(records), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.enumerate()

    def load(index: tf.Tensor, item: tuple[tf.Tensor, tf.Tensor]):
        path, label = item
        image = decode_image(path)
        if training:
            image = augment_image(image, index, seed)
        return image, label

    dataset = dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    options = tf.data.Options()
    options.deterministic = True
    return dataset.with_options(options)


def build_model(image_size: tuple[int, int] = IMAGE_SIZE) -> keras.Model:
    """Build the grayscale CNN architecture used in the source notebook."""
    inputs = keras.Input(shape=(*image_size, IMAGE_CHANNELS), name="grayscale_pixels")
    x = keras.layers.Rescaling(1.0 / 255.0, name="normalize")(inputs)
    x = keras.layers.Conv2D(16, 7, strides=2, padding="same", activation="relu", name="conv1")(x)
    x = keras.layers.MaxPooling2D(2, strides=2, name="pool1")(x)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="conv2")(x)
    x = keras.layers.MaxPooling2D(2, strides=2, name="pool2")(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv3")(x)
    x = keras.layers.MaxPooling2D(2, strides=2, name="pool3")(x)
    x = keras.layers.Flatten(name="flatten")(x)
    x = keras.layers.Dense(224, activation="relu", name="classifier_dense")(x)
    x = keras.layers.Dropout(0.2, name="classifier_dropout")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", name="ok_probability")(x)
    model = keras.Model(inputs, outputs, name="casting_defect_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    predictions = (scores >= DECISION_THRESHOLD).astype(np.int32)
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    accuracy = (tp + tn) / max(1, len(labels))
    precision_ok = tp / max(1, tp + fp)
    recall_ok = tp / max(1, tp + fn)
    precision_defect = tn / max(1, tn + fn)
    recall_defect = tn / max(1, tn + fp)
    return {
        "accuracy": float(accuracy),
        "precision_ok": float(precision_ok),
        "recall_ok": float(recall_ok),
        "precision_defect": float(precision_defect),
        "recall_defect": float(recall_defect),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "sample_count": int(len(labels)),
    }


def select_accuracy_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Choose an accuracy-maximizing threshold using validation data only."""
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape or labels.size == 0:
        raise ValueError("Validation labels and scores must be non-empty and equally sized")
    unique_scores = np.unique(scores)
    if unique_scores.size == 1:
        return DECISION_THRESHOLD
    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    candidates = np.concatenate(([0.0], midpoints, [1.0]))
    accuracies = np.asarray(
        [np.mean((scores >= threshold).astype(np.int32) == labels) for threshold in candidates]
    )
    best = candidates[np.isclose(accuracies, np.max(accuracies), rtol=0.0, atol=1e-12)]
    return float(best[np.argmin(np.abs(best - DECISION_THRESHOLD))])


def calibrate_model_threshold(model: keras.Model, source_threshold: float) -> float:
    """Move a source probability threshold to 0.5 by shifting final-layer logits."""
    if not 0.0 < source_threshold < 1.0:
        raise ValueError("Calibration threshold must be strictly between zero and one")
    output_layer = model.get_layer("ok_probability")
    if not isinstance(output_layer, keras.layers.Dense):
        raise TypeError("Expected ok_probability to be a Dense layer")
    weights = output_layer.get_weights()
    if len(weights) != 2 or weights[1].shape != (1,):
        raise ValueError("Expected a single-output Dense layer with a bias")
    logit_threshold = float(np.log(source_threshold / (1.0 - source_threshold)))
    kernel, bias = weights
    output_layer.set_weights([kernel, bias - logit_threshold])
    return -logit_threshold


def evaluate_keras_model(
    model: keras.Model, records: Sequence[ImageRecord], batch_size: int, seed: int
) -> tuple[dict[str, object], np.ndarray]:
    dataset = make_dataset(records, batch_size, training=False, seed=seed)
    scores = model.predict(dataset, verbose=0).reshape(-1)
    labels = np.asarray([record.label for record in records], dtype=np.int32)
    return binary_metrics(labels, scores), scores


def balanced_sample(records: Sequence[ImageRecord], limit: int, seed: int) -> tuple[ImageRecord, ...]:
    rng = np.random.default_rng(seed)
    per_class = max(1, limit // len(CLASS_NAMES))
    chosen: list[ImageRecord] = []
    for label in range(len(CLASS_NAMES)):
        candidates = [record for record in records if record.label == label]
        indices = rng.permutation(len(candidates))[:per_class]
        chosen.extend(candidates[index] for index in indices)
    return tuple(chosen[:limit])


def representative_dataset(records: Sequence[ImageRecord]) -> Callable[[], Iterator[list[np.ndarray]]]:
    def generate() -> Iterator[list[np.ndarray]]:
        yield [np.zeros((1, *IMAGE_SIZE, IMAGE_CHANNELS), dtype=np.float32)]
        yield [np.full((1, *IMAGE_SIZE, IMAGE_CHANNELS), 255.0, dtype=np.float32)]
        for record in records:
            image = decode_image(record.path).numpy().astype(np.float32)
            yield [image[np.newaxis, ...]]

    return generate


def _json_tensor_details(details: dict) -> dict:
    scale, zero_point = details["quantization"]
    return {
        "name": details["name"],
        "shape": [int(value) for value in details["shape"]],
        "dtype": np.dtype(details["dtype"]).name,
        "scale": float(scale),
        "zero_point": int(zero_point),
    }


def inspect_tflite_model(model_content: bytes) -> tuple[dict, dict, tuple[str, ...]]:
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    operators = tuple(
        sorted({item["op_name"] for item in interpreter._get_ops_details()})  # noqa: SLF001
    )
    return _json_tensor_details(input_details), _json_tensor_details(output_details), operators


def convert_to_tflite(
    model: keras.Model,
    representative_data: Callable[[], Iterable[list[np.ndarray]]],
    output_path: Path,
) -> ConversionResult:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    model_content = converter.convert()
    input_details, output_details, operators = inspect_tflite_model(model_content)
    if input_details["dtype"] != "uint8" or output_details["dtype"] != "uint8":
        raise ValueError(
            f"Expected uint8 boundaries, got {input_details['dtype']} -> {output_details['dtype']}"
        )
    expected_shape = [1, *[int(value) for value in model.input_shape[1:]]]
    if input_details["shape"] != expected_shape:
        raise ValueError(f"Unexpected TFLite input shape: {input_details['shape']}")
    output_path.write_bytes(model_content)
    return ConversionResult(output_path, input_details, output_details, operators)


def quantize_for_tensor(values: np.ndarray, details: dict) -> np.ndarray:
    scale, zero_point = details["quantization"]
    dtype = details["dtype"]
    if scale == 0:
        return values.astype(dtype)
    quantized = np.round(values / scale + zero_point)
    bounds = np.iinfo(dtype)
    return np.clip(quantized, bounds.min, bounds.max).astype(dtype)


def dequantize_tensor(values: np.ndarray, details: dict) -> np.ndarray:
    scale, zero_point = details["quantization"]
    if scale == 0:
        return values.astype(np.float32)
    return (values.astype(np.float32) - zero_point) * scale


def evaluate_tflite(
    model_path: Path, records: Sequence[ImageRecord]
) -> tuple[dict[str, object], np.ndarray]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    scores: list[float] = []
    for index, record in enumerate(records, start=1):
        image = decode_image(record.path).numpy()[np.newaxis, ...]
        interpreter.set_tensor(input_details["index"], quantize_for_tensor(image, input_details))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])
        scores.append(float(dequantize_tensor(output, output_details).reshape(-1)[0]))
        if index % 100 == 0 or index == len(records):
            print(f"TFLite evaluation: {index}/{len(records)}", flush=True)
    labels = np.asarray([record.label for record in records], dtype=np.int32)
    score_array = np.asarray(scores, dtype=np.float32)
    return binary_metrics(labels, score_array), score_array


def write_metadata(
    output_dir: Path,
    splits: DatasetSplits,
    history: dict[str, list[float]],
    keras_metrics: dict[str, object],
    tflite_metrics: dict[str, object],
    conversion: ConversionResult,
    keras_model_path: Path,
    parameter_count: int,
    calibration: dict[str, object],
    validation_metrics: dict[str, object],
) -> dict:
    accuracy_drop = float(keras_metrics["accuracy"] - tflite_metrics["accuracy"])
    metadata = {
        "model": {
            "name": "casting_defect_cnn",
            "architecture": "notebook_style_custom_cnn",
            "uses_transfer_learning": False,
            "parameters": parameter_count,
            "keras_model_bytes": keras_model_path.stat().st_size,
            "tflite_model_bytes": conversion.model_path.stat().st_size,
        },
        "dataset": {"splits": split_summary(splits), "deduplication_audit": splits.audit},
        "training": {
            "augmentation": {"zoom_factor": [0.9, 1.0], "brightness_factor": [0.9, 1.0]},
            "history": history,
            "decision_boundary_calibration": calibration,
        },
        "input": {
            "shape": [1, *IMAGE_SIZE, IMAGE_CHANNELS],
            "color_space": "grayscale",
            "resize": list(IMAGE_SIZE),
            "resize_location": "Android caller before inference",
            "dtype": conversion.input_details["dtype"],
            "scale": conversion.input_details["scale"],
            "zero_point": conversion.input_details["zero_point"],
            "raw_value_range": [0, 255],
        },
        "output": {
            "shape": [1, 1],
            "meaning": "probability of ok_front after dequantization",
            "threshold": DECISION_THRESHOLD,
            "below_threshold": "def_front",
            "at_or_above_threshold": "ok_front",
            "dtype": conversion.output_details["dtype"],
            "scale": conversion.output_details["scale"],
            "zero_point": conversion.output_details["zero_point"],
        },
        "classes": CLASS_TO_INDEX,
        "conversion": {
            "mode": "full_int8_builtins",
            "uses_select_tf_ops": False,
            "operators": list(conversion.operators),
            "android_dependency": "LiteRT built-in runtime",
        },
        "metrics": {
            "validation_after_calibration": validation_metrics,
            "keras": keras_metrics,
            "tflite": tflite_metrics,
            "accuracy_drop": accuracy_drop,
        },
        "versions": {
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
        },
    }
    (output_dir / "cnn_metrics.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def enforce_acceptance(metadata: dict, allow_low_accuracy: bool) -> bool:
    tflite_accuracy = float(metadata["metrics"]["tflite"]["accuracy"])
    accuracy_drop = float(metadata["metrics"]["accuracy_drop"])
    failures: list[str] = []
    if tflite_accuracy < MINIMUM_TFLITE_ACCURACY:
        failures.append(
            f"TFLite accuracy {tflite_accuracy:.4f} is below {MINIMUM_TFLITE_ACCURACY:.4f}"
        )
    if accuracy_drop > MAXIMUM_QUANTIZATION_ACCURACY_DROP + 1e-9:
        failures.append(
            f"Quantization accuracy drop {accuracy_drop:.4f} exceeds "
            f"{MAXIMUM_QUANTIZATION_ACCURACY_DROP:.4f}"
        )
    if not failures:
        return True
    message = "; ".join(failures)
    if allow_low_accuracy:
        print(f"WARNING: {message}", file=sys.stderr)
        return False
    raise RuntimeError(message)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    set_reproducibility(args.seed)
    configure_tensorflow_memory_growth()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Auditing and splitting the dataset...", flush=True)
    splits = discover_dataset(
        args.data_dir,
        args.validation_fraction,
        args.seed,
        args.max_images_per_split,
    )
    print(json.dumps(split_summary(splits), indent=2), flush=True)
    print(json.dumps(splits.audit, indent=2), flush=True)

    train_dataset = make_dataset(splits.train, args.batch_size, training=True, seed=args.seed)
    validation_dataset = make_dataset(
        splits.validation, args.batch_size, training=False, seed=args.seed
    )
    model = build_model()
    parameter_count = model.count_params()
    model.summary()

    checkpoint_path = args.output_dir / "casting_defect_cnn_best.weights.h5"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=2, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path, monitor="val_loss", save_best_only=True, save_weights_only=True
        ),
    ]
    history_object = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        shuffle=False,
        verbose=2,
    )
    history = {
        key: [float(value) for value in values] for key, values in history_object.history.items()
    }
    if checkpoint_path.exists():
        model.load_weights(checkpoint_path)

    uncalibrated_validation_metrics, validation_scores = evaluate_keras_model(
        model, splits.validation, args.batch_size, args.seed
    )
    validation_labels = np.asarray([record.label for record in splits.validation])
    source_threshold = select_accuracy_threshold(validation_labels, validation_scores)
    bias_adjustment = calibrate_model_threshold(model, source_threshold)
    validation_metrics, _ = evaluate_keras_model(
        model, splits.validation, args.batch_size, args.seed
    )
    calibration = {
        "selected_using": "validation split only",
        "uncalibrated_threshold": source_threshold,
        "final_model_threshold": DECISION_THRESHOLD,
        "final_layer_bias_adjustment": bias_adjustment,
        "uncalibrated_validation_metrics_at_0_5": uncalibrated_validation_metrics,
    }
    print(f"Decision-boundary calibration: {json.dumps(calibration, indent=2)}", flush=True)

    keras_model_path = args.output_dir / "casting_defect_cnn.keras"
    model.save(keras_model_path)
    (args.output_dir / "labels.txt").write_text("def_front\nok_front\n")
    (args.output_dir / "cnn_training_history.json").write_text(
        json.dumps(history, indent=2) + "\n"
    )

    keras_metrics, _ = evaluate_keras_model(model, splits.test, args.batch_size, args.seed)
    print(f"Keras test metrics: {json.dumps(keras_metrics, indent=2)}", flush=True)

    calibration_records = balanced_sample(
        splits.train, args.representative_samples, args.seed
    )
    tflite_path = args.output_dir / "casting_defect_cnn_int8.tflite"
    print("Converting to built-in-only full-INT8 TFLite...", flush=True)
    conversion = convert_to_tflite(
        model, representative_dataset(calibration_records), tflite_path
    )
    tflite_metrics, _ = evaluate_tflite(tflite_path, splits.test)
    print(f"TFLite test metrics: {json.dumps(tflite_metrics, indent=2)}", flush=True)

    metadata = write_metadata(
        args.output_dir,
        splits,
        history,
        keras_metrics,
        tflite_metrics,
        conversion,
        keras_model_path,
        parameter_count,
        calibration,
        validation_metrics,
    )
    accepted = enforce_acceptance(metadata, args.allow_low_accuracy)
    print(f"Wrote {tflite_path} ({tflite_path.stat().st_size:,} bytes)", flush=True)
    print(f"Acceptance gates passed: {accepted}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
