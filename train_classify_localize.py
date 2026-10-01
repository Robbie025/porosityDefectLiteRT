#!/usr/bin/env python3
"""Train a notebook-inspired binary classifier with an approximate spatial map.

The casting dataset supplies image labels only. The second output is therefore
class evidence learned from those labels, not a supervised defect mask.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import keras
import numpy as np
import tensorflow as tf
from PIL import Image

from train_tflite_cnn import (
    IMAGE_SIZE,
    ImageRecord,
    augment_image,
    balanced_sample,
    binary_metrics,
    decode_image,
    dequantize_tensor,
    discover_dataset,
    make_dataset,
    quantize_for_tensor,
    representative_dataset,
    select_accuracy_threshold,
    set_reproducibility,
    split_summary,
)


MIN_ACCURACY = 0.98
MAX_ACCURACY_DROP = 0.01


def build_models(image_size: tuple[int, int] = IMAGE_SIZE) -> tuple[keras.Model, keras.Model]:
    """Keep the notebook classifier and add an auxiliary spatial classifier."""
    inputs = keras.Input(shape=(*image_size, 1), name="grayscale_pixels")
    x = keras.layers.Rescaling(1.0 / 255.0, name="normalize")(inputs)
    x = keras.layers.Conv2D(16, 7, strides=2, padding="same", activation="relu", name="conv1")(x)
    x = keras.layers.MaxPooling2D(2, strides=2, name="pool1")(x)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="conv2")(x)
    x = keras.layers.MaxPooling2D(2, strides=2, name="pool2")(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv3")(x)
    x = keras.layers.MaxPooling2D(2, strides=2, name="pool3")(x)
    dense = keras.layers.Flatten(name="flatten")(x)
    dense = keras.layers.Dense(224, activation="relu", name="classifier_dense")(dense)
    dense = keras.layers.Dropout(0.2, name="classifier_dropout")(dense)
    ok_probability = keras.layers.Dense(1, activation="sigmoid", name="ok_probability")(dense)
    # The auxiliary image loss trains a spatial map from the same weak labels.
    ok_logits = keras.layers.Conv2D(1, 1, name="ok_logit_map")(x)
    pooled = keras.layers.GlobalAveragePooling2D(name="pooled_ok_logit")(ok_logits)
    cam_probability = keras.layers.Activation("sigmoid", name="cam_probability")(pooled)
    negative_logits = keras.layers.Rescaling(-1.0, name="negative_ok_logits")(ok_logits)
    defect_heatmap = keras.layers.Activation("sigmoid", name="defect_heatmap")(
        negative_logits
    )
    trainer = keras.Model(inputs, [ok_probability, cam_probability], name="casting_cam_trainer")
    exporter = keras.Model(
        inputs, [ok_probability, defect_heatmap], name="casting_cam_export"
    )
    trainer.compile(
        optimizer=keras.optimizers.Adam(),
        loss={"ok_probability": keras.losses.BinaryCrossentropy(),
              "cam_probability": keras.losses.BinaryCrossentropy()},
        loss_weights={"ok_probability": 1.0, "cam_probability": 0.2},
        metrics={"ok_probability": [keras.metrics.BinaryAccuracy(name="accuracy")]},
    )
    return trainer, exporter


def calibrate_threshold(model: keras.Model, threshold: float) -> float:
    """Move a validation-selected OK threshold to the exported 0.5 boundary."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("Calibration threshold must be strictly between zero and one")
    layer = model.get_layer("ok_probability")
    kernel, bias = layer.get_weights()
    adjustment = -float(np.log(threshold / (1.0 - threshold)))
    layer.set_weights([kernel, bias + adjustment])
    return adjustment


def keras_scores(model: keras.Model, records: Sequence[ImageRecord], batch_size: int) -> np.ndarray:
    data = make_dataset(records, batch_size, training=False, seed=0)
    scores = model.predict(data, verbose=0)
    if isinstance(scores, list):
        scores = scores[0]
    return np.asarray(scores, np.float32).reshape(-1)


def make_multitask_dataset(records: Sequence[ImageRecord], batch_size: int, *, training: bool, seed: int) -> tf.data.Dataset:
    base = make_dataset(records, batch_size, training=training, seed=seed)
    return base.map(lambda image, label: (image, {"ok_probability": label, "cam_probability": label}))


def initialize_from_existing(trainer: keras.Model, source_path: Path | None) -> bool:
    if source_path is None or not source_path.is_file():
        return False
    previous = keras.models.load_model(source_path, compile=False)
    for name in ("conv1", "conv2", "conv3", "classifier_dense", "ok_probability"):
        trainer.get_layer(name).set_weights(previous.get_layer(name).get_weights())
    return True


def _tensor_info(details: dict) -> dict:
    scale, zero_point = details["quantization"]
    return {
        "name": str(details["name"]),
        "shape": [int(value) for value in details["shape"]],
        "dtype": np.dtype(details["dtype"]).name,
        "scale": float(scale),
        "zero_point": int(zero_point),
    }


def identify_outputs(outputs: list[dict]) -> tuple[dict, dict]:
    """TFLite can reorder outputs; identify them by their unambiguous shapes."""
    scores = [item for item in outputs if tuple(item["shape"]) == (1, 1)]
    maps = [
        item for item in outputs
        if len(item["shape"]) == 4 and item["shape"][0] == 1
        and item["shape"][-1] == 1 and item["shape"][1] > 1
        and item["shape"][2] > 1
    ]
    if len(scores) != 1 or len(maps) != 1 or len(outputs) != 2:
        raise ValueError(f"Expected scalar score and spatial heatmap, got {outputs}")
    return scores[0], maps[0]


def convert_model(model: keras.Model, samples, path: Path) -> dict:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = samples
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    content = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=content)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or tuple(inputs[0]["shape"]) != (1, *IMAGE_SIZE, 1):
        raise ValueError("Unexpected TFLite input tensor")
    score, heatmap = identify_outputs(outputs)
    if any(np.dtype(item["dtype"]) != np.dtype(np.uint8) for item in [inputs[0], score, heatmap]):
        raise ValueError("All public TFLite tensors must be uint8")
    operators = sorted({item["op_name"] for item in interpreter._get_ops_details()})
    if any(item.startswith("Flex") for item in operators):
        raise ValueError("TFLite model requires unsupported Flex operators")
    path.write_bytes(content)
    return {
        "input": _tensor_info(inputs[0]),
        "ok_probability": _tensor_info(score),
        "defect_heatmap": _tensor_info(heatmap),
        "operators": operators,
    }


def evaluate_tflite(path: Path, records: Sequence[ImageRecord]) -> tuple[dict, np.ndarray]:
    interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    score_details, _ = identify_outputs(interpreter.get_output_details())
    scores = []
    for index, record in enumerate(records, 1):
        image = decode_image(record.path).numpy()[None, ...]
        interpreter.set_tensor(input_details["index"], quantize_for_tensor(image, input_details))
        interpreter.invoke()
        raw = interpreter.get_tensor(score_details["index"])
        scores.append(float(dequantize_tensor(raw, score_details).reshape(-1)[0]))
        if index % 100 == 0 or index == len(records):
            print(f"TFLite evaluation: {index}/{len(records)}", flush=True)
    values = np.asarray(scores, np.float32)
    labels = np.asarray([record.label for record in records], np.int32)
    return binary_metrics(labels, values), values


def save_sample_overlays(model_path: Path, records: Sequence[ImageRecord], output_dir: Path) -> None:
    """Save a few visual sanity checks, one per class when available."""
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    source = interpreter.get_input_details()[0]
    _, heat = identify_outputs(interpreter.get_output_details())
    chosen = [next(record for record in records if record.label == label) for label in (0, 1)]
    for record in chosen:
        image = decode_image(record.path).numpy().astype(np.uint8)
        interpreter.set_tensor(source["index"], quantize_for_tensor(image[None, ...], source))
        interpreter.invoke()
        evidence = dequantize_tensor(interpreter.get_tensor(heat["index"]), heat)[0, :, :, 0]
        evidence = tf.image.resize(evidence[..., None], IMAGE_SIZE).numpy()[:, :, 0]
        baseline = float(np.median(evidence))
        peak = float(np.max(evidence))
        normalized = np.clip((evidence - baseline) / max(peak - baseline, 1e-6), 0.0, 1.0)
        normalized[normalized < 0.55] = 0.0
        rgb = np.repeat(image, 3, axis=2).astype(np.float32)
        rgb[:, :, 0] = rgb[:, :, 0] * (1 - 0.5 * normalized) + 255 * 0.5 * normalized
        rgb[:, :, 1:] *= (1 - 0.5 * normalized[:, :, None])
        Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(
            output_dir / f"example_{'defect' if record.label == 0 else 'ok'}.png"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/casting_data/casting_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/classify_localize"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--representative-samples", type=int, default=200)
    parser.add_argument("--max-images-per-split", type=int)
    parser.add_argument("--init-from", type=Path, default=Path("artifacts/casting_defect_cnn.keras"),
                        help="Existing notebook-style classifier weights; skipped when absent")
    parser.add_argument("--allow-low-accuracy", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.representative_samples < 2:
        parser.error("epochs, batch size, and representative samples must be positive")
    if not 0 < args.validation_fraction < 0.5:
        parser.error("validation fraction must be between 0 and 0.5")
    if args.max_images_per_split is not None and args.max_images_per_split < 2:
        parser.error("max images per split must be at least 2")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    set_reproducibility(args.seed)
    splits = discover_dataset(args.data_dir, args.validation_fraction, args.seed, args.max_images_per_split)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(split_summary(splits), indent=2), flush=True)
    trainer, exporter = build_models()
    initialized = initialize_from_existing(trainer, args.init_from)
    print(f"Initialized notebook classification layers from {args.init_from}: {initialized}", flush=True)
    history = trainer.fit(
        make_multitask_dataset(splits.train, args.batch_size, training=True, seed=args.seed),
        validation_data=make_multitask_dataset(splits.validation, args.batch_size, training=False, seed=args.seed),
        epochs=args.epochs,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        verbose=2,
    ).history
    validation_labels = np.asarray([record.label for record in splits.validation], np.int32)
    raw_validation = keras_scores(trainer, splits.validation, args.batch_size)
    threshold = select_accuracy_threshold(validation_labels, raw_validation)
    adjustment = calibrate_threshold(trainer, threshold)
    calibrated_validation = keras_scores(trainer, splits.validation, args.batch_size)
    keras_model_path = args.output_dir / "casting_classify_localize.keras"
    tflite_path = args.output_dir / "casting_classify_localize_int8.tflite"
    exporter.save(keras_model_path)
    contract = convert_model(
        exporter,
        representative_dataset(balanced_sample(splits.train, args.representative_samples, args.seed)),
        tflite_path,
    )
    keras_test_scores = keras_scores(trainer, splits.test, args.batch_size)
    test_labels = np.asarray([record.label for record in splits.test], np.int32)
    keras_test = binary_metrics(test_labels, keras_test_scores)
    tflite_test, _ = evaluate_tflite(tflite_path, splits.test)
    accuracy_drop = keras_test["accuracy"] - tflite_test["accuracy"]
    result = {
        "classes": {"def_front": 0, "ok_front": 1},
        "dataset": {"splits": split_summary(splits), "audit": splits.audit},
        "model": {"parameters": trainer.count_params(), "architecture": "notebook_classifier_with_auxiliary_class_activation_map", "initialized_from_existing_classifier": initialized},
        "contract": contract,
        "calibration": {"validation_source_threshold": threshold, "logit_adjustment": adjustment, "exported_threshold": 0.5},
        "metrics": {
            "validation": binary_metrics(validation_labels, calibrated_validation),
            "keras_test": keras_test,
            "tflite_test": tflite_test,
            "accuracy_drop": accuracy_drop,
        },
        "localization": "Auxiliary weakly supervised evidence map; camera rectangles use occlusion sensitivity of the primary classifier because no ground-truth boxes or masks are available",
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "history.json").write_text(json.dumps({k: [float(v) for v in values] for k, values in history.items()}, indent=2) + "\n")
    (args.output_dir / "labels.txt").write_text("def_front\nok_front\n")
    save_sample_overlays(tflite_path, splits.test, args.output_dir)
    print(json.dumps(result["metrics"], indent=2), flush=True)
    failures = []
    if tflite_test["accuracy"] < MIN_ACCURACY:
        failures.append(f"TFLite test accuracy {tflite_test['accuracy']:.3f} below {MIN_ACCURACY:.2f}")
    if accuracy_drop > MAX_ACCURACY_DROP:
        failures.append(f"quantization accuracy drop {accuracy_drop:.3f} above {MAX_ACCURACY_DROP:.2f}")
    if failures:
        print("; ".join(failures), file=sys.stderr)
        return 0 if args.allow_low_accuracy else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
