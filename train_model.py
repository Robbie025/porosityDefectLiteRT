#!/usr/bin/env python3
"""Train and export a MobileNetV5 casting-defect classifier.

The 294M-parameter backbone remains frozen. Its pooled features are cached so
that the small binary classification head can be trained on modest hardware.
The resulting composed model is converted to a uint8 TensorFlow Lite model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import keras
import keras_hub
import numpy as np
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


PRESET = "mobilenetv5_300m_enc_gemma3n"
CLASS_NAMES = ("def_front", "ok_front")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_SIZE = (224, 224)
FEATURE_DIM = 2048
DECISION_THRESHOLD = 0.5
EXPECTED_V5_PARAMETERS = 294_284_096
ESTIMATED_INT8_BYTES = EXPECTED_V5_PARAMETERS
ESTIMATED_FP32_BYTES = EXPECTED_V5_PARAMETERS * 4


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
    conversion_mode: str
    used_select_tf_ops: bool
    input_details: dict
    output_details: dict
    operators: tuple[str, ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frozen MobileNetV5 casting-defect classifier and export TFLite."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("datasets/casting_data/casting_data"),
        help="Directory containing train/ and test/ class directories.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--stage",
        choices=("all", "train", "convert"),
        default="all",
        help=(
            "Run both isolated stages, only feature/head training and SavedModel export, "
            "or only TFLite conversion/evaluation."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--head-batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--representative-samples",
        type=int,
        default=200,
        help="Maximum balanced samples used for TFLite calibration.",
    )
    parser.add_argument(
        "--force-rebuild-cache",
        action="store_true",
        help="Discard compatible cached backbone features and extract them again.",
    )
    parser.add_argument(
        "--allow-low-accuracy",
        action="store_true",
        help="Keep artifacts instead of failing when the 98%% accuracy gate is missed.",
    )
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        default=None,
        help="Development-only cap applied per class and split.",
    )
    args = parser.parse_args(argv)
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("--validation-fraction must be between 0 and 0.5")
    for name in ("feature_batch_size", "head_batch_size", "epochs", "representative_samples"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_images_per_split is not None and args.max_images_per_split <= 0:
        parser.error("--max-images-per-split must be positive")
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
        rng = np.random.default_rng(seed + label)
        indices = rng.permutation(len(class_records))
        validation_count = max(1, int(round(len(class_records) * validation_fraction)))
        validation_indices = set(indices[:validation_count].tolist())
        validation.extend(record for i, record in enumerate(class_records) if i in validation_indices)
        train.extend(record for i, record in enumerate(class_records) if i not in validation_indices)
    return tuple(sorted(train, key=lambda item: str(item.path))), tuple(
        sorted(validation, key=lambda item: str(item.path))
    )


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
            continue
        if test_label != record.label:
            raise ValueError(
                f"Identical image has conflicting train/test labels: {record.path}"
            )
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
        records = getattr(splits, split_name)
        counts = {name: 0 for name in CLASS_NAMES}
        for record in records:
            counts[CLASS_NAMES[record.label]] += 1
        counts["total"] = len(records)
        summary[split_name] = counts
    return summary


def decode_image(path: Path, image_size: tuple[int, int] = IMAGE_SIZE) -> tf.Tensor:
    encoded = tf.io.read_file(str(path))
    image = tf.io.decode_jpeg(encoded, channels=3)
    image = tf.image.resize(image, image_size, method="bilinear", antialias=True)
    return tf.clip_by_value(tf.cast(image, tf.float32), 0.0, 255.0)


def image_variant(image: tf.Tensor, variant: str) -> tf.Tensor:
    if variant == "original":
        return image
    if variant == "zoom":
        height, width = IMAGE_SIZE
        crop_height = int(round(height * 0.9))
        crop_width = int(round(width * 0.9))
        cropped = tf.image.resize_with_crop_or_pad(image, crop_height, crop_width)
        return tf.image.resize(cropped, IMAGE_SIZE, method="bilinear", antialias=True)
    if variant == "brightness":
        return tf.clip_by_value(image * 0.9, 0.0, 255.0)
    raise ValueError(f"Unknown image variant: {variant}")


def load_image_batch(records: Sequence[ImageRecord], variants: Sequence[str]) -> tf.Tensor:
    images = [image_variant(decode_image(record.path), variant) for record, variant in zip(records, variants)]
    return tf.stack(images)


def build_mobilenetv5_feature_extractor(
    preset: str = PRESET,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> tuple[keras.Model, keras.Model, keras.layers.Layer]:
    print(f"Loading KerasHub preset {preset!r}. The initial download is approximately 1.2 GB.")
    preset_converter = keras_hub.layers.MobileNetV5ImageConverter.from_preset(preset)
    converter_config = preset_converter.get_config()
    preset_scale = np.asarray(converter_config["scale"], dtype=np.float32)
    preset_offset = np.asarray(converter_config["offset"], dtype=np.float32)
    if not np.allclose(preset_scale, 1.0 / 255.0) or not np.allclose(preset_offset, 0.0):
        raise ValueError(
            "The MobileNetV5 preset preprocessing changed; review its scale and offset "
            "before exporting the Android model."
        )
    # Input is already exactly 224x224. Keeping KerasHub's bicubic resize in the
    # graph emits tf.ScaleAndTranslate, which neither built-in TFLite nor Flex
    # can lower. Rescaling alone is mathematically identical at this fixed size.
    preprocessor = keras.layers.Rescaling(
        scale=1.0 / 255.0,
        offset=0.0,
        name="mobilenetv5_rescale",
    )
    backbone = keras_hub.models.MobileNetV5Backbone.from_preset(preset)
    backbone.trainable = False

    raw_input = keras.Input(shape=(*image_size, 3), dtype="float32", name="rgb_image")
    prepared = preprocessor(raw_input)
    feature_map = backbone(prepared, training=False)
    if len(feature_map.shape) == 4:
        pooled = keras.layers.GlobalAveragePooling2D(name="global_average_pool")(feature_map)
    elif len(feature_map.shape) == 2:
        pooled = feature_map
    else:
        raise ValueError(f"Unexpected MobileNetV5 output shape: {feature_map.shape}")
    feature_model = keras.Model(raw_input, pooled, name="mobilenetv5_feature_extractor")
    if feature_model.output_shape[-1] != FEATURE_DIM:
        raise ValueError(
            f"Expected {FEATURE_DIM} pooled features, got {feature_model.output_shape[-1]}"
        )
    return feature_model, backbone, preprocessor


def build_classification_head(feature_dim: int = FEATURE_DIM) -> keras.Model:
    features = keras.Input(shape=(feature_dim,), dtype="float32", name="pooled_features")
    x = keras.layers.Dense(224, activation="relu", name="classifier_dense")(features)
    x = keras.layers.Dropout(0.2, name="classifier_dropout")(x)
    probability = keras.layers.Dense(1, activation="sigmoid", name="ok_probability")(x)
    return keras.Model(features, probability, name="casting_defect_head")


def compose_inference_model(feature_model: keras.Model, head: keras.Model) -> keras.Model:
    raw_input = keras.Input(shape=(*IMAGE_SIZE, 3), dtype="float32", name="image")
    probability = head(feature_model(raw_input, training=False), training=False)
    return keras.Model(raw_input, probability, name="casting_defect_mobilenetv5")


def dataset_fingerprint(records: Sequence[ImageRecord], variants: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(PRESET.encode())
    digest.update(str(IMAGE_SIZE).encode())
    for record in records:
        stat = record.path.stat()
        digest.update(str(record.path.resolve()).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    for variant in variants:
        digest.update(variant.encode())
    return digest.hexdigest()


def _cache_paths(cache_dir: Path, split_name: str) -> tuple[Path, Path, Path]:
    return (
        cache_dir / f"{split_name}_features.npy",
        cache_dir / f"{split_name}_labels.npy",
        cache_dir / f"{split_name}_cache.json",
    )


def extract_or_load_features(
    feature_model: keras.Model,
    records: Sequence[ImageRecord],
    variants_per_record: Sequence[str],
    cache_dir: Path,
    split_name: str,
    batch_size: int,
    force_rebuild: bool,
) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    features_path, labels_path, metadata_path = _cache_paths(cache_dir, split_name)
    fingerprint = dataset_fingerprint(records, variants_per_record)
    expected_count = len(records) * len(variants_per_record)
    if not force_rebuild and features_path.exists() and labels_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("fingerprint") == fingerprint and metadata.get("count") == expected_count:
            print(f"Using cached {split_name} features from {features_path}")
            return np.load(features_path, mmap_mode="r"), np.load(labels_path, mmap_mode="r")

    features = np.lib.format.open_memmap(
        features_path,
        mode="w+",
        dtype=np.float32,
        shape=(expected_count, FEATURE_DIM),
    )
    labels = np.lib.format.open_memmap(
        labels_path,
        mode="w+",
        dtype=np.float32,
        shape=(expected_count,),
    )
    expanded: list[tuple[ImageRecord, str]] = [
        (record, variant) for record in records for variant in variants_per_record
    ]
    started = time.monotonic()
    for offset in range(0, expected_count, batch_size):
        batch_items = expanded[offset : offset + batch_size]
        batch_records = [item[0] for item in batch_items]
        batch_variants = [item[1] for item in batch_items]
        batch_images = load_image_batch(batch_records, batch_variants)
        batch_features = feature_model(batch_images, training=False).numpy()
        end = offset + len(batch_items)
        features[offset:end] = batch_features
        labels[offset:end] = [record.label for record in batch_records]
        if offset == 0 or end == expected_count or end % max(batch_size * 25, 25) == 0:
            elapsed = time.monotonic() - started
            rate = end / elapsed if elapsed else 0.0
            print(f"{split_name}: {end}/{expected_count} features ({rate:.2f} images/s)")
    features.flush()
    labels.flush()
    metadata_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "count": expected_count,
                "feature_dim": FEATURE_DIM,
                "variants": list(variants_per_record),
            },
            indent=2,
        )
        + "\n"
    )
    return np.load(features_path, mmap_mode="r"), np.load(labels_path, mmap_mode="r")


def train_head(
    head: keras.Model,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    output_dir: Path,
    batch_size: int,
    epochs: int,
    seed: int,
) -> dict[str, list[float]]:
    head.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    checkpoint_path = output_dir / "best_head.weights.h5"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_features, train_labels))
        .shuffle(len(train_labels), seed=seed, reshuffle_each_iteration=True)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    validation_dataset = (
        tf.data.Dataset.from_tensor_slices((validation_features, validation_labels))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    history = head.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    if checkpoint_path.exists():
        head.load_weights(checkpoint_path)
    return {key: [float(value) for value in values] for key, values in history.history.items()}


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    predictions = (scores >= DECISION_THRESHOLD).astype(np.int32)
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    accuracy = (tp + tn) / max(1, len(labels))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "accuracy": float(accuracy),
        "precision_ok": float(precision),
        "recall_ok": float(recall),
        "f1_ok": float(f1),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "sample_count": int(len(labels)),
    }


def evaluate_head(head: keras.Model, features: np.ndarray, labels: np.ndarray, batch_size: int) -> dict:
    scores = head.predict(features, batch_size=batch_size, verbose=0).reshape(-1)
    return binary_metrics(labels, scores)


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
        # Anchor the model boundary to the documented raw uint8 range. Without
        # these endpoints, a dark calibration subset can yield an input scale
        # slightly below 1.0, making direct Android byte input less intuitive.
        yield [np.zeros((1, *IMAGE_SIZE, 3), dtype=np.float32)]
        yield [np.full((1, *IMAGE_SIZE, 3), 255.0, dtype=np.float32)]
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
    model: keras.Model | Path,
    representative_data: Callable[[], Iterable[list[np.ndarray]]],
    output_path: Path,
) -> ConversionResult:
    def summarize_error(error: Exception, limit: int = 2_000) -> str:
        message = str(error)
        if len(message) <= limit:
            return message
        return message[:limit] + "\n... converter diagnostic truncated ..."

    def converter_for(allow_select_ops: bool) -> tf.lite.TFLiteConverter:
        if isinstance(model, Path):
            if not model.is_dir():
                raise FileNotFoundError(f"SavedModel directory does not exist: {model}")
            converter = tf.lite.TFLiteConverter.from_saved_model(str(model))
        else:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data
        # Per-channel quantization causes the 294M-parameter MLIR graph to
        # exceed 15 GiB RAM plus 4 GiB swap on the reference workstation.
        # Per-tensor weights remain full INT8 and substantially reduce peak
        # converter memory; the post-conversion accuracy gate protects quality.
        converter._experimental_disable_per_channel = True  # noqa: SLF001
        converter._experimental_disable_per_channel_quantization_for_dense_layers = (  # noqa: SLF001
            True
        )
        operations = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        if allow_select_ops:
            operations.append(tf.lite.OpsSet.SELECT_TF_OPS)
        converter.target_spec.supported_ops = operations
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
        return converter

    try:
        print("Converting with built-in INT8 operators only...")
        model_content = converter_for(False).convert()
        conversion_mode = "full_int8_builtins_per_tensor"
        used_select_tf_ops = False
    except Exception as builtins_error:  # Converter error types vary by TensorFlow release.
        print(
            f"Built-in-only conversion failed: {summarize_error(builtins_error)}",
            file=sys.stderr,
        )
        print("Retrying with SELECT_TF_OPS enabled...", file=sys.stderr)
        try:
            model_content = converter_for(True).convert()
        except Exception as select_error:
            raise RuntimeError(
                "MobileNetV5 conversion failed with both built-in INT8 and SELECT_TF_OPS"
            ) from select_error
        conversion_mode = "int8_per_tensor_with_select_tf_ops"
        used_select_tf_ops = True

    input_details, output_details, operators = inspect_tflite_model(model_content)
    if input_details["dtype"] != "uint8" or output_details["dtype"] != "uint8":
        raise ValueError(
            f"Expected uint8 model boundaries, got {input_details['dtype']} -> {output_details['dtype']}"
        )
    output_path.write_bytes(model_content)
    return ConversionResult(
        model_path=output_path,
        conversion_mode=conversion_mode,
        used_select_tf_ops=used_select_tf_ops,
        input_details=input_details,
        output_details=output_details,
        operators=operators,
    )


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


def evaluate_tflite(model_path: Path, records: Sequence[ImageRecord]) -> tuple[dict, np.ndarray]:
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
            print(f"TFLite evaluation: {index}/{len(records)}")
    labels = np.asarray([record.label for record in records], dtype=np.int32)
    score_array = np.asarray(scores, dtype=np.float32)
    return binary_metrics(labels, score_array), score_array


def write_metadata(
    output_dir: Path,
    splits: DatasetSplits,
    keras_metrics: dict,
    tflite_metrics: dict,
    conversion: ConversionResult,
    keras_model_path: Path,
) -> dict:
    accuracy_delta = float(keras_metrics["accuracy"] - tflite_metrics["accuracy"])
    metadata = {
        "model": {
            "name": "casting_defect_mobilenetv5",
            "backbone": PRESET,
            "backbone_frozen": True,
            "backbone_parameters": EXPECTED_V5_PARAMETERS,
            "keras_model_bytes": keras_model_path.stat().st_size,
            "tflite_model_bytes": conversion.model_path.stat().st_size,
        },
        "mobilenetv5_feasibility": {
            "decision": "selected_for_pixel_8_pro",
            "target_device": "Google Pixel 8 Pro (12 GB RAM, Tensor G3)",
            "estimated_fp32_weight_bytes": ESTIMATED_FP32_BYTES,
            "estimated_int8_weight_bytes": ESTIMATED_INT8_BYTES,
            "estimated_android_inference_ram_bytes": [400_000_000, 1_000_000_000],
            "training_strategy": "frozen backbone with cached pooled features",
            "caveats": [
                "large application asset",
                "Pixel NPU acceleration is not assumed",
                "SELECT_TF_OPS may cause CPU fallback",
                "not intended for real-time video",
            ],
        },
        "dataset": {"splits": split_summary(splits), "deduplication_audit": splits.audit},
        "input": {
            "shape": [1, *IMAGE_SIZE, 3],
            "color_order": "RGB",
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
            "mode": conversion.conversion_mode,
            "weight_granularity": "per_tensor",
            "uses_select_tf_ops": conversion.used_select_tf_ops,
            "operators": list(conversion.operators),
            "android_dependency": (
                "LiteRT plus SELECT_TF_OPS runtime"
                if conversion.used_select_tf_ops
                else "LiteRT built-in runtime"
            ),
        },
        "metrics": {
            "keras": keras_metrics,
            "tflite": tflite_metrics,
            "accuracy_drop": accuracy_delta,
        },
        "versions": {
            "python": sys.version.split()[0],
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "keras_hub": keras_hub.__version__,
            "numpy": np.__version__,
        },
    }
    (output_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def enforce_acceptance(metadata: dict, allow_low_accuracy: bool) -> bool:
    tflite_accuracy = float(metadata["metrics"]["tflite"]["accuracy"])
    accuracy_drop = float(metadata["metrics"]["accuracy_drop"])
    failures: list[str] = []
    if tflite_accuracy < 0.98:
        failures.append(f"TFLite accuracy {tflite_accuracy:.4f} is below 0.9800")
    if accuracy_drop > 0.01 + 1e-9:
        failures.append(f"Quantization accuracy drop {accuracy_drop:.4f} exceeds 0.0100")
    if failures:
        message = "; ".join(failures)
        if allow_low_accuracy:
            print(f"WARNING: {message}", file=sys.stderr)
            return False
        else:
            raise RuntimeError(message)
    return True


def run_training_stage(
    args: argparse.Namespace, splits: DatasetSplits, output_dir: Path
) -> tuple[dict, Path]:
    feature_model, _, _ = build_mobilenetv5_feature_extractor()
    cache_dir = output_dir / "feature_cache"
    train_features, train_labels = extract_or_load_features(
        feature_model,
        splits.train,
        ("original", "zoom", "brightness"),
        cache_dir,
        "train",
        args.feature_batch_size,
        args.force_rebuild_cache,
    )
    validation_features, validation_labels = extract_or_load_features(
        feature_model,
        splits.validation,
        ("original",),
        cache_dir,
        "validation",
        args.feature_batch_size,
        args.force_rebuild_cache,
    )
    test_features, test_labels = extract_or_load_features(
        feature_model,
        splits.test,
        ("original",),
        cache_dir,
        "test",
        args.feature_batch_size,
        args.force_rebuild_cache,
    )

    head = build_classification_head()
    history = train_head(
        head,
        train_features,
        train_labels,
        validation_features,
        validation_labels,
        output_dir,
        args.head_batch_size,
        args.epochs,
        args.seed,
    )
    (output_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    keras_metrics = evaluate_head(head, test_features, test_labels, args.head_batch_size)
    (output_dir / "keras_test_metrics.json").write_text(
        json.dumps(keras_metrics, indent=2) + "\n"
    )
    print(f"Keras test metrics: {json.dumps(keras_metrics, indent=2)}")

    inference_model = compose_inference_model(feature_model, head)
    keras_model_path = output_dir / "casting_defect_mobilenetv5.keras"
    inference_model.save(keras_model_path)
    (output_dir / "labels.txt").write_text("def_front\nok_front\n")
    saved_model_path = output_dir / "saved_model"
    if saved_model_path.exists():
        shutil.rmtree(saved_model_path)
    inference_model.export(saved_model_path, format="tf_saved_model", verbose=False)
    return keras_metrics, saved_model_path


def run_conversion_stage(
    args: argparse.Namespace, splits: DatasetSplits, output_dir: Path
) -> dict:
    saved_model_path = output_dir / "saved_model"
    keras_model_path = output_dir / "casting_defect_mobilenetv5.keras"
    keras_metrics_path = output_dir / "keras_test_metrics.json"
    if not keras_model_path.is_file() or not keras_metrics_path.is_file():
        raise FileNotFoundError(
            "Training artifacts are missing. Run train_model.py --stage train before conversion."
        )
    keras_metrics = json.loads(keras_metrics_path.read_text())

    calibration_records = balanced_sample(splits.train, args.representative_samples, args.seed)
    conversion = convert_to_tflite(
        saved_model_path,
        representative_dataset(calibration_records),
        output_dir / "casting_defect_mobilenetv5_int8.tflite",
    )
    tflite_metrics, _ = evaluate_tflite(conversion.model_path, splits.test)
    print(f"TFLite test metrics: {json.dumps(tflite_metrics, indent=2)}")
    metadata = write_metadata(
        output_dir,
        splits,
        keras_metrics,
        tflite_metrics,
        conversion,
        keras_model_path,
    )
    accepted = enforce_acceptance(metadata, args.allow_low_accuracy)
    if accepted:
        print(f"Android model ready: {conversion.model_path}")
    else:
        print(
            f"Smoke artifact generated but acceptance gates were bypassed: {conversion.model_path}"
        )
    return metadata


def conversion_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        "convert",
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(args.output_dir),
        "--seed",
        str(args.seed),
        "--validation-fraction",
        str(args.validation_fraction),
        "--representative-samples",
        str(args.representative_samples),
    ]
    if args.max_images_per_split is not None:
        command.extend(["--max-images-per-split", str(args.max_images_per_split)])
    if args.allow_low_accuracy:
        command.append("--allow-low-accuracy")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    set_reproducibility(args.seed)
    configure_tensorflow_memory_growth()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = discover_dataset(
        args.data_dir,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        max_images_per_split=args.max_images_per_split,
    )
    print(json.dumps({"splits": split_summary(splits), "audit": splits.audit}, indent=2))

    if args.stage in ("all", "train"):
        run_training_stage(args, splits, output_dir)
        if args.stage == "train":
            print(
                "Training/export stage complete. Run the same command with --stage convert "
                "after ensuring ample free system RAM."
            )
            return 0
        # Replacing the process is intentional: TensorFlow's CPU allocator may
        # retain the full backbone after clear_session(), which causes the
        # 294M-parameter MLIR conversion to be killed by the OOM manager.
        command = conversion_command(args)
        print("Restarting in an isolated TFLite conversion process...")
        sys.stdout.flush()
        os.execv(sys.executable, command)

    run_conversion_stage(args, splits, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
