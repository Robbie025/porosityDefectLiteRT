#!/usr/bin/env python3
"""Live USB-camera test for the casting-defect TFLite model.

The classifier has no spatial output, so localization uses occlusion
sensitivity: blur one region at a time and measure whether the image becomes
more likely to be OK. The resulting circles are explanations, not measured
defect boundaries.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np
import tensorflow as tf

try:
    import cv2
except ImportError as error:  # pragma: no cover - exercised only without the dependency.
    raise SystemExit(
        "OpenCV is required. Install it in the existing environment with:\n"
        "  .venv/bin/pip install opencv-python"
    ) from error


MODEL_SIZE = 300
MODEL_SHAPE = (1, MODEL_SIZE, MODEL_SIZE, 1)
DECISION_THRESHOLD = 0.5
SCORE_WINDOW = 5
DEFECT_CONFIRMATION_FRAMES = 3
OCCLUSION_GRID_SIZE = 6
OCCLUSION_PATCH_SIZE = 75
MIN_HOTSPOT_SENSITIVITY = 0.01
HOTSPOT_RELATIVE_THRESHOLD = 0.5
MAX_HOTSPOTS = 3
DARK_RED = (0, 0, 139)
GREEN = (0, 180, 0)
WHITE = (245, 245, 245)
YELLOW = (0, 210, 255)
WINDOW_NAME = "Casting porosity TFLite camera test"


@dataclass(frozen=True)
class CropRegion:
    x: int
    y: int
    size: int


@dataclass(frozen=True)
class Hotspot:
    x: float
    y: float
    radius: float
    sensitivity: float


@dataclass(frozen=True)
class FrameHotspot:
    x: int
    y: int
    radius: int
    sensitivity: float


@dataclass(frozen=True)
class CameraSelection:
    capture: cv2.VideoCapture
    index: int
    first_frame: np.ndarray


class TFLiteClassifier:
    """Validated uint8 wrapper around the casting classifier."""

    def __init__(self, model_path: Path, threads: int) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"TFLite model does not exist: {model_path}")
        self.model_path = model_path
        self.interpreter = tf.lite.Interpreter(
            model_path=str(model_path), num_threads=threads
        )
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        outputs = self.interpreter.get_output_details()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                f"Expected one input and one output, got {len(inputs)} and {len(outputs)}"
            )
        self.input = inputs[0]
        self.output = outputs[0]
        actual_shape = tuple(int(value) for value in self.input["shape"])
        output_shape = tuple(int(value) for value in self.output["shape"])
        if actual_shape != MODEL_SHAPE:
            raise ValueError(f"Expected model input {MODEL_SHAPE}, got {actual_shape}")
        if output_shape != (1, 1):
            raise ValueError(f"Expected model output (1, 1), got {output_shape}")
        if np.dtype(self.input["dtype"]) != np.dtype(np.uint8):
            raise ValueError(f"Expected uint8 model input, got {self.input['dtype']}")
        if np.dtype(self.output["dtype"]) != np.dtype(np.uint8):
            raise ValueError(f"Expected uint8 model output, got {self.output['dtype']}")

    @staticmethod
    def _quantize(values: np.ndarray, details: dict) -> np.ndarray:
        scale, zero_point = details["quantization"]
        dtype = details["dtype"]
        if scale == 0:
            return values.astype(dtype)
        quantized = np.rint(values.astype(np.float32) / scale + zero_point)
        bounds = np.iinfo(dtype)
        return np.clip(quantized, bounds.min, bounds.max).astype(dtype)

    @staticmethod
    def _dequantize(values: np.ndarray, details: dict) -> np.ndarray:
        scale, zero_point = details["quantization"]
        if scale == 0:
            return values.astype(np.float32)
        return (values.astype(np.float32) - zero_point) * scale

    def predict_ok(self, grayscale: np.ndarray) -> tuple[float, float]:
        if grayscale.shape != (MODEL_SIZE, MODEL_SIZE):
            raise ValueError(
                f"Expected a {MODEL_SIZE}x{MODEL_SIZE} grayscale image, got {grayscale.shape}"
            )
        model_input = grayscale[np.newaxis, :, :, np.newaxis]
        quantized = self._quantize(model_input, self.input)
        started = time.perf_counter()
        self.interpreter.set_tensor(self.input["index"], quantized)
        self.interpreter.invoke()
        elapsed_ms = (time.perf_counter() - started) * 1_000.0
        raw_output = self.interpreter.get_tensor(self.output["index"])
        ok_score = float(self._dequantize(raw_output, self.output).reshape(-1)[0])
        return float(np.clip(ok_score, 0.0, 1.0)), elapsed_ms


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show a USB-camera preview, classify casting defects, approximate "
            "hotspots, and save five annotated defect results."
        )
    )
    parser.add_argument(
        "--camera-index",
        default="auto",
        help="Camera number or 'auto' to probe available video devices.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/casting_defect_cnn_int8.tflite"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("read_images"))
    parser.add_argument("--max-saved-images", type=int, default=5)
    parser.add_argument("--save-interval", type=float, default=2.0)
    parser.add_argument("--terminal-interval", type=float, default=5.0)
    parser.add_argument("--localization-interval", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Validate inference, localization, rendering, and save limits without a camera.",
    )
    args = parser.parse_args(argv)
    if args.camera_index != "auto":
        try:
            args.camera_index = int(args.camera_index)
        except ValueError:
            parser.error("--camera-index must be a non-negative integer or 'auto'")
        if args.camera_index < 0:
            parser.error("--camera-index must be non-negative")
    for name in ("max_saved_images", "threads", "width", "height"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    for name in ("save_interval", "terminal_interval", "localization_interval"):
        if getattr(args, name) <= 0.0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def prepare_frame(frame: np.ndarray) -> tuple[np.ndarray, CropRegion]:
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Camera frame must be a non-empty BGR image")
    height, width = frame.shape[:2]
    side = min(height, width)
    x = (width - side) // 2
    y = (height - side) // 2
    square = frame[y : y + side, x : x + side]
    gray = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
    interpolation = cv2.INTER_AREA if side >= MODEL_SIZE else cv2.INTER_LINEAR
    resized = cv2.resize(gray, (MODEL_SIZE, MODEL_SIZE), interpolation=interpolation)
    return np.ascontiguousarray(resized, dtype=np.uint8), CropRegion(x, y, side)


def localize_defect(
    classifier: TFLiteClassifier,
    grayscale: np.ndarray,
    baseline_ok_score: float,
) -> tuple[list[Hotspot], float]:
    """Return influential model-space regions and total localization latency."""
    started = time.perf_counter()
    blurred = cv2.GaussianBlur(grayscale, (31, 31), sigmaX=0)
    last_origin = MODEL_SIZE - OCCLUSION_PATCH_SIZE
    positions = np.linspace(0, last_origin, OCCLUSION_GRID_SIZE).round().astype(int)
    candidates: list[Hotspot] = []
    radius = OCCLUSION_PATCH_SIZE / 2.0
    for y in positions:
        for x in positions:
            occluded = grayscale.copy()
            occluded[y : y + OCCLUSION_PATCH_SIZE, x : x + OCCLUSION_PATCH_SIZE] = blurred[
                y : y + OCCLUSION_PATCH_SIZE, x : x + OCCLUSION_PATCH_SIZE
            ]
            occluded_ok, _ = classifier.predict_ok(occluded)
            # Blurring an influential defective region should increase P(OK).
            sensitivity = max(0.0, occluded_ok - baseline_ok_score)
            candidates.append(
                Hotspot(x + radius, y + radius, radius, float(sensitivity))
            )

    maximum = max((candidate.sensitivity for candidate in candidates), default=0.0)
    threshold = max(
        MIN_HOTSPOT_SENSITIVITY, maximum * HOTSPOT_RELATIVE_THRESHOLD
    )
    eligible = sorted(
        (candidate for candidate in candidates if candidate.sensitivity >= threshold),
        key=lambda candidate: candidate.sensitivity,
        reverse=True,
    )
    selected: list[Hotspot] = []
    minimum_distance = OCCLUSION_PATCH_SIZE * 0.75
    for candidate in eligible:
        if all(
            np.hypot(candidate.x - chosen.x, candidate.y - chosen.y) >= minimum_distance
            for chosen in selected
        ):
            selected.append(candidate)
        if len(selected) == MAX_HOTSPOTS:
            break
    return selected, (time.perf_counter() - started) * 1_000.0


def map_hotspots_to_frame(
    hotspots: Sequence[Hotspot], crop: CropRegion
) -> list[FrameHotspot]:
    scale = crop.size / MODEL_SIZE
    mapped: list[FrameHotspot] = []
    for hotspot in hotspots:
        mapped.append(
            FrameHotspot(
                x=int(round(crop.x + hotspot.x * scale)),
                y=int(round(crop.y + hotspot.y * scale)),
                radius=max(4, int(round(hotspot.radius * scale))),
                sensitivity=hotspot.sensitivity,
            )
        )
    return mapped


def _outlined_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.65,
    thickness: int = 2,
) -> None:
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_analysis(
    frame: np.ndarray,
    crop: CropRegion,
    *,
    confirmed_defect: bool,
    possible_defect: bool,
    ok_score: float,
    inference_ms: float,
    fps: float,
    hotspots: Sequence[FrameHotspot],
    location_uncertain: bool,
    saved_count: int,
    max_saved_images: int,
    result_number: int | None = None,
    captured_at: datetime | None = None,
) -> np.ndarray:
    annotated = frame.copy()
    cv2.rectangle(
        annotated,
        (crop.x, crop.y),
        (crop.x + crop.size - 1, crop.y + crop.size - 1),
        YELLOW,
        2,
    )
    _outlined_text(
        annotated,
        "Inspection region",
        (crop.x + 8, max(24, crop.y + 25)),
        YELLOW,
        scale=0.55,
    )

    if confirmed_defect:
        state_text = "DEFECT DETECTED"
        state_color = DARK_RED
    elif possible_defect:
        state_text = "CHECKING POSSIBLE DEFECT"
        state_color = YELLOW
    else:
        state_text = "OK"
        state_color = GREEN
    _outlined_text(annotated, state_text, (20, 38), state_color, scale=0.9, thickness=2)
    _outlined_text(
        annotated,
        f"OK {ok_score:.1%} | defect {1.0 - ok_score:.1%}",
        (20, 70),
        WHITE,
        scale=0.65,
    )
    _outlined_text(
        annotated,
        f"Inference {inference_ms:.1f} ms | preview {fps:.1f} FPS",
        (20, 100),
        WHITE,
        scale=0.55,
    )
    _outlined_text(
        annotated,
        f"Saved results: {saved_count}/{max_saved_images}",
        (20, 128),
        WHITE,
        scale=0.55,
    )

    if confirmed_defect:
        if hotspots:
            for index, hotspot in enumerate(hotspots, start=1):
                cv2.circle(
                    annotated,
                    (hotspot.x, hotspot.y),
                    hotspot.radius,
                    DARK_RED,
                    3,
                    cv2.LINE_AA,
                )
                label_x = max(5, hotspot.x - hotspot.radius)
                label_y = max(22, hotspot.y - hotspot.radius - 7)
                _outlined_text(
                    annotated,
                    f"Approx. hotspot {index}",
                    (label_x, label_y),
                    DARK_RED,
                    scale=0.5,
                )
        elif location_uncertain:
            center = (crop.x + crop.size // 2, crop.y + crop.size // 2)
            cv2.circle(
                annotated,
                center,
                max(5, crop.size // 2 - 5),
                DARK_RED,
                3,
                cv2.LINE_AA,
            )
            _outlined_text(
                annotated,
                "DEFECT - location uncertain",
                (20, 158),
                DARK_RED,
                scale=0.6,
            )
        _outlined_text(
            annotated,
            "Approximate localization - not a measured boundary",
            (20, annotated.shape[0] - 20),
            DARK_RED,
            scale=0.55,
        )

    if result_number is not None:
        when = captured_at or datetime.now().astimezone()
        _outlined_text(
            annotated,
            f"Analysis result {result_number}/{max_saved_images}",
            (20, 188),
            DARK_RED,
            scale=0.7,
        )
        _outlined_text(
            annotated,
            when.isoformat(timespec="milliseconds"),
            (20, 216),
            WHITE,
            scale=0.5,
        )
    return annotated


def result_path(
    output_dir: Path, result_number: int, captured_at: datetime
) -> Path:
    timestamp = captured_at.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    base = output_dir / f"analysis_{timestamp}_{result_number:02d}.jpg"
    if not base.exists():
        return base
    collision = 1
    while True:
        candidate = output_dir / (
            f"analysis_{timestamp}_{result_number:02d}_{collision:02d}.jpg"
        )
        if not candidate.exists():
            return candidate
        collision += 1


def write_result(path: Path, image: np.ndarray) -> bool:
    return bool(cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 95]))


def _camera_candidates(camera_index: str | int) -> list[int]:
    if isinstance(camera_index, int):
        return [camera_index]
    video_devices = sorted(
        Path("/dev").glob("video*"),
        key=lambda path: int(path.name.removeprefix("video"))
        if path.name.removeprefix("video").isdigit()
        else 10_000,
    )
    candidates = [
        int(path.name.removeprefix("video"))
        for path in video_devices
        if path.name.removeprefix("video").isdigit()
    ]
    return candidates or list(range(10))


def _open_capture(index: int, width: int, height: int) -> cv2.VideoCapture | None:
    capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        return None
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def select_camera(camera_index: str | int, width: int, height: int) -> CameraSelection:
    attempted: list[int] = []
    for index in _camera_candidates(camera_index):
        attempted.append(index)
        capture = _open_capture(index, width, height)
        if capture is None:
            continue
        frame = None
        for _ in range(5):
            success, candidate = capture.read()
            if success and candidate is not None and candidate.size:
                frame = candidate
                break
            time.sleep(0.05)
        if frame is not None:
            return CameraSelection(capture, index, frame)
        capture.release()
    attempted_text = ", ".join(str(index) for index in attempted) or "none"
    raise RuntimeError(f"No camera returned a frame; attempted indices: {attempted_text}")


def run_self_test(args: argparse.Namespace) -> int:
    classifier = TFLiteClassifier(args.model, args.threads)
    height, width = 720, 1280
    frame = np.full((height, width, 3), 175, dtype=np.uint8)
    cv2.circle(frame, (width // 2, height // 2), 250, (205, 205, 205), -1)
    cv2.circle(frame, (width // 2, height // 2), 120, (40, 40, 40), -1)
    cv2.circle(frame, (width // 2 + 130, height // 2 - 90), 18, (15, 15, 15), -1)
    grayscale, crop = prepare_frame(frame)
    if grayscale.shape != (MODEL_SIZE, MODEL_SIZE) or grayscale.dtype != np.uint8:
        raise AssertionError("Preprocessing contract failed")
    ok_score, inference_ms = classifier.predict_ok(grayscale)
    hotspots, localization_ms = localize_defect(classifier, grayscale, ok_score)
    synthetic_hotspots = hotspots or [Hotspot(210.0, 75.0, 37.5, 0.1)]
    mapped = map_hotspots_to_frame(synthetic_hotspots, crop)
    for hotspot in mapped:
        if not (0 <= hotspot.x < width and 0 <= hotspot.y < height and hotspot.radius > 0):
            raise AssertionError(f"Mapped hotspot is outside the frame: {hotspot}")

    names: set[str] = set()
    for number in range(1, args.max_saved_images + 1):
        captured_at = datetime.now().astimezone()
        name = result_path(args.output_dir, number, captured_at).name
        if name in names:
            raise AssertionError(f"Generated duplicate result filename: {name}")
        names.add(name)
        annotated = draw_analysis(
            frame,
            crop,
            confirmed_defect=True,
            possible_defect=True,
            ok_score=min(ok_score, 0.49),
            inference_ms=inference_ms,
            fps=30.0,
            hotspots=mapped,
            location_uncertain=not bool(mapped),
            saved_count=number,
            max_saved_images=args.max_saved_images,
            result_number=number,
            captured_at=captured_at,
        )
        encoded, buffer = cv2.imencode(
            ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 95]
        )
        if not encoded or buffer.size == 0:
            raise AssertionError("Annotated JPEG encoding failed")

    print(
        "Self-test passed: "
        f"model={args.model}, ok_score={ok_score:.4f}, "
        f"inference={inference_ms:.2f} ms, localization={localization_ms:.2f} ms, "
        f"hotspots={len(hotspots)}, rendered_results={args.max_saved_images}"
    )
    return 0


def run_camera(args: argparse.Namespace) -> int:
    classifier = TFLiteClassifier(args.model, args.threads)
    selection = select_camera(args.camera_index, args.width, args.height)
    capture = selection.capture
    frame = selection.first_frame
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Using camera /dev/video{selection.index}; model={args.model}; "
        f"saving up to {args.max_saved_images} results in {args.output_dir}"
    )
    print("Press Q or Esc to exit.")

    scores: deque[float] = deque(maxlen=SCORE_WINDOW)
    consecutive_defects = 0
    hotspots: list[FrameHotspot] = []
    location_uncertain = False
    last_localization_at = float("-inf")
    localization_ms = 0.0
    last_save_at = float("-inf")
    saved_count = 0
    last_saved_path: Path | None = None
    last_report_at = time.monotonic()
    frames_since_report = 0
    fps = 0.0
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            if frame is None:
                success, frame = capture.read()
                if not success or frame is None:
                    raise RuntimeError(f"Camera /dev/video{selection.index} stopped returning frames")
            now = time.monotonic()
            grayscale, crop = prepare_frame(frame)
            raw_ok_score, inference_ms = classifier.predict_ok(grayscale)
            scores.append(raw_ok_score)
            smoothed_ok = float(np.median(np.asarray(scores, dtype=np.float32)))
            possible_defect = smoothed_ok < DECISION_THRESHOLD
            consecutive_defects = consecutive_defects + 1 if possible_defect else 0
            confirmed_defect = consecutive_defects >= DEFECT_CONFIRMATION_FRAMES
            localized_this_frame = False

            if confirmed_defect and now - last_localization_at >= args.localization_interval:
                model_hotspots, localization_ms = localize_defect(
                    classifier, grayscale, raw_ok_score
                )
                hotspots = map_hotspots_to_frame(model_hotspots, crop)
                location_uncertain = not bool(hotspots)
                last_localization_at = now
                localized_this_frame = True
            elif not confirmed_defect:
                hotspots = []
                location_uncertain = False
                last_localization_at = float("-inf")

            if (
                confirmed_defect
                and localized_this_frame
                and saved_count < args.max_saved_images
                and now - last_save_at >= args.save_interval
            ):
                result_number = saved_count + 1
                captured_at = datetime.now().astimezone()
                annotated_result = draw_analysis(
                    frame,
                    crop,
                    confirmed_defect=True,
                    possible_defect=True,
                    ok_score=smoothed_ok,
                    inference_ms=inference_ms,
                    fps=fps,
                    hotspots=hotspots,
                    location_uncertain=location_uncertain,
                    saved_count=result_number,
                    max_saved_images=args.max_saved_images,
                    result_number=result_number,
                    captured_at=captured_at,
                )
                path = result_path(args.output_dir, result_number, captured_at)
                if write_result(path, annotated_result):
                    saved_count = result_number
                    last_save_at = now
                    last_saved_path = path
                else:
                    print(f"ERROR: Failed to save annotated result: {path}", file=sys.stderr)

            frames_since_report += 1
            report_elapsed = now - last_report_at
            if report_elapsed >= args.terminal_interval:
                fps = frames_since_report / report_elapsed
                state = "DEFECT" if confirmed_defect else ("CHECKING" if possible_defect else "OK")
                latest_path = str(last_saved_path) if last_saved_path else "none"
                print(
                    f"{datetime.now().astimezone().isoformat(timespec='seconds')} "
                    f"camera=/dev/video{selection.index} "
                    f"resolution={frame.shape[1]}x{frame.shape[0]} fps={fps:.1f} "
                    f"ok={smoothed_ok:.3f} defect={1.0-smoothed_ok:.3f} state={state} "
                    f"inference_ms={inference_ms:.1f} localization_ms={localization_ms:.1f} "
                    f"hotspots={len(hotspots)} saved={saved_count}/{args.max_saved_images} "
                    f"latest={latest_path}",
                    flush=True,
                )
                last_report_at = now
                frames_since_report = 0

            display = draw_analysis(
                frame,
                crop,
                confirmed_defect=confirmed_defect,
                possible_defect=possible_defect,
                ok_score=smoothed_ok,
                inference_ms=inference_ms,
                fps=fps,
                hotspots=hotspots,
                location_uncertain=location_uncertain,
                saved_count=saved_count,
                max_saved_images=args.max_saved_images,
            )
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            try:
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            frame = None
    finally:
        capture.release()
        cv2.destroyAllWindows()

    print(
        f"Camera test stopped. Saved {saved_count}/{args.max_saved_images} "
        f"annotated defect results in {args.output_dir}."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test(args)
    return run_camera(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Camera test interrupted.")
        raise SystemExit(130) from None
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from None
