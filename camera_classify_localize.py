#!/usr/bin/env python3
"""USB-camera preview for the two-output casting classifier.

Type 1 then Enter in the terminal to begin inference. Type 0 then Enter to
stop inference, close the preview, and exit.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Thread
import time
from typing import Sequence

import cv2
import numpy as np
import tensorflow as tf


MODEL_SIZE = 300
WINDOW = "Porosity classification and approximate localization"
RED = (0, 0, 139)
ALERT_RED = (0, 0, 255)
GREEN = (0, 180, 0)
YELLOW = (0, 210, 255)


@dataclass(frozen=True)
class Region:
    x: int
    y: int
    side: int


def focus_region(frame: np.ndarray) -> Region:
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 2:
        raise ValueError("Expected a nonempty BGR camera frame")
    height, width = frame.shape[:2]
    side = min(height, width)
    return Region((width - side) // 2, (height - side) // 2, side)


def prepare_frame(frame: np.ndarray) -> tuple[np.ndarray, Region]:
    region = focus_region(frame)
    crop = frame[region.y:region.y + region.side, region.x:region.x + region.side]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    interpolation = cv2.INTER_AREA if region.side >= MODEL_SIZE else cv2.INTER_LINEAR
    pixels = cv2.resize(gray, (MODEL_SIZE, MODEL_SIZE), interpolation=interpolation)
    return np.ascontiguousarray(pixels, np.uint8), region


def _quantize(values: np.ndarray, details: dict) -> np.ndarray:
    scale, zero_point = details["quantization"]
    if scale <= 0:
        raise ValueError("TFLite tensor has invalid quantization scale")
    dtype = np.dtype(details["dtype"])
    bounds = np.iinfo(dtype)
    return np.clip(np.rint(values.astype(np.float32) / scale + zero_point), bounds.min, bounds.max).astype(dtype)


def _dequantize(values: np.ndarray, details: dict) -> np.ndarray:
    scale, zero_point = details["quantization"]
    if scale <= 0:
        raise ValueError("TFLite tensor has invalid quantization scale")
    return (values.astype(np.float32) - zero_point) * scale


def identify_outputs(outputs: list[dict]) -> tuple[dict, dict]:
    score = [x for x in outputs if tuple(x["shape"]) == (1, 1)]
    heat = [
        x for x in outputs
        if len(x["shape"]) == 4 and x["shape"][0] == 1
        and x["shape"][-1] == 1 and x["shape"][1] > 1 and x["shape"][2] > 1
    ]
    if len(outputs) != 2 or len(score) != 1 or len(heat) != 1:
        raise ValueError("Model must provide one scalar OK score and one spatial heatmap")
    return score[0], heat[0]


class Classifier:
    def __init__(self, path: Path, threads: int = 4) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"TFLite model not found: {path}")
        self.interpreter = tf.lite.Interpreter(model_path=str(path), num_threads=threads)
        self.interpreter.allocate_tensors()
        inputs = self.interpreter.get_input_details()
        if len(inputs) != 1 or tuple(inputs[0]["shape"]) != (1, MODEL_SIZE, MODEL_SIZE, 1):
            raise ValueError("Expected a 300x300 grayscale TFLite input")
        self.source = inputs[0]
        self.score, self.heat = identify_outputs(self.interpreter.get_output_details())
        for details in (self.source, self.score, self.heat):
            if np.dtype(details["dtype"]) != np.dtype(np.uint8):
                raise ValueError("Expected uint8 input and output tensors")

    def predict(self, pixels: np.ndarray) -> tuple[float, np.ndarray]:
        if pixels.shape != (MODEL_SIZE, MODEL_SIZE) or pixels.dtype != np.uint8:
            raise ValueError("Expected 300x300 uint8 grayscale pixels")
        data = _quantize(pixels[None, :, :, None], self.source)
        self.interpreter.set_tensor(self.source["index"], data)
        self.interpreter.invoke()
        probability = float(_dequantize(self.interpreter.get_tensor(self.score["index"]), self.score)[0, 0])
        heatmap = _dequantize(self.interpreter.get_tensor(self.heat["index"]), self.heat)[0, :, :, 0]
        return float(np.clip(probability, 0, 1)), np.clip(heatmap, 0, 1)


def heatmap_boxes(
    heatmap: np.ndarray,
    region: Region,
    *,
    relative_threshold: float = 0.75,
    absolute_threshold: float = 0.55,
    min_area_fraction: float = 0.002,
) -> list[tuple[int, int, int, int, float]]:
    """Turn spatial evidence into approximate frame-space rectangles."""
    if heatmap.ndim != 2 or not np.all(np.isfinite(heatmap)):
        raise ValueError("Heatmap must be a finite two-dimensional array")
    smooth = cv2.resize(heatmap.astype(np.float32), (region.side, region.side), interpolation=cv2.INTER_LINEAR)
    smooth = cv2.GaussianBlur(smooth, (0, 0), sigmaX=max(1, region.side / 75))
    peak = float(smooth.max())
    threshold = max(absolute_threshold, peak * relative_threshold)
    if peak < threshold:
        return []
    mask = (smooth >= threshold).astype(np.uint8)
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    minimum_area = max(1, int(region.side * region.side * min_area_fraction))
    boxes = []
    for component in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[component]]
        if area < minimum_area:
            continue
        evidence = float(smooth[components == component].max())
        boxes.append((region.x + x, region.y + y, region.x + x + width - 1, region.y + y + height - 1, evidence))
    return sorted(boxes, key=lambda box: box[4], reverse=True)[:3]


def occlusion_boxes(
    classifier: Classifier,
    pixels: np.ndarray,
    region: Region,
    baseline_ok: float,
    *,
    grid_size: int = 6,
    patch_size: int = 75,
) -> list[tuple[int, int, int, int, float]]:
    """Locate regions whose blur makes the classifier less sure of a defect.

    This directly explains the exported classification score. It is a coarse
    model sensitivity test, not a measured pore boundary.
    """
    blurred = cv2.GaussianBlur(pixels, (31, 31), sigmaX=0)
    origins = np.linspace(0, MODEL_SIZE - patch_size, grid_size).round().astype(int)
    candidates = []
    for y in origins:
        for x in origins:
            altered = pixels.copy()
            altered[y:y + patch_size, x:x + patch_size] = blurred[y:y + patch_size, x:x + patch_size]
            occluded_ok, _ = classifier.predict(altered)
            sensitivity = max(0.0, occluded_ok - baseline_ok)
            candidates.append((int(x), int(y), sensitivity))
    strongest = max((item[2] for item in candidates), default=0.0)
    if strongest < 0.01:
        return []
    selected = []
    for x, y, sensitivity in sorted(candidates, key=lambda item: item[2], reverse=True):
        if sensitivity < max(0.01, strongest * 0.5):
            break
        if any(np.hypot(x - old_x, y - old_y) < patch_size * 0.75 for old_x, old_y, _ in selected):
            continue
        selected.append((x, y, sensitivity))
        if len(selected) == 3:
            break
    scale = region.side / MODEL_SIZE
    return [
        (
            region.x + int(round(x * scale)),
            region.y + int(round(y * scale)),
            region.x + min(region.side - 1, int(round((x + patch_size) * scale))),
            region.y + min(region.side - 1, int(round((y + patch_size) * scale))),
            sensitivity,
        )
        for x, y, sensitivity in selected
    ]


def blink_visible(now: float, *, period: float = 0.25) -> bool:
    return int(now / period) % 2 == 0


def draw_preview(
    frame: np.ndarray,
    region: Region,
    *,
    active: bool,
    now: float,
    ok_score: float | None = None,
    boxes: Sequence[tuple[int, int, int, int, float]] = (),
) -> np.ndarray:
    output = frame.copy()
    defect = active and ok_score is not None and ok_score < 0.5
    flash = defect and blink_visible(now)
    if not active or blink_visible(now):
        cv2.rectangle(output, (region.x, region.y), (region.x + region.side - 1, region.y + region.side - 1), ALERT_RED if flash else YELLOW, 3)
    for x1, y1, x2, y2, evidence in boxes:
        cv2.rectangle(output, (x1, y1), (x2, y2), RED, 2)
        cv2.putText(output, f"Approx. sensitivity {evidence:.2f}", (max(4, x1), max(20, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 2, cv2.LINE_AA)
    if flash:
        cv2.rectangle(output, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), ALERT_RED, 10)
    cv2.rectangle(output, (0, 0), (frame.shape[1] - 1, min(78, frame.shape[0] - 1)), ALERT_RED if flash else (25, 25, 25), -1)
    if not active:
        state = "READY: type 1 then Enter in terminal"
        color = YELLOW
    elif ok_score is None:
        state = "STARTING INFERENCE"
        color = YELLOW
    elif ok_score < 0.5:
        state = f"NOT OK: DEFECT {1 - ok_score:.1%}"
        color = (255, 255, 255) if flash else ALERT_RED
    else:
        state = f"OK {ok_score:.1%}"
        color = GREEN
    cv2.putText(output, state, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    if active:
        cv2.putText(output, "type 0 then Enter to exit", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2, cv2.LINE_AA)
    return output


def terminal_reader(commands: Queue[str], stream) -> None:
    for line in stream:
        value = line.strip()
        if value in ("0", "1"):
            commands.put(value)
        else:
            print("Enter 1 to start or 0 to stop.", flush=True)
    commands.put("0")


def apply_commands(commands: Queue[str], active: bool) -> tuple[bool, bool]:
    """Return (active, exit_requested), draining commands in arrival order."""
    try:
        while True:
            command = commands.get_nowait()
            if command == "0":
                return False, True
            if command == "1":
                active = True
    except Empty:
        return active, False


def camera_candidates(camera_index: str) -> list[int]:
    if camera_index != "auto":
        return [int(camera_index)]
    found = sorted(
        int(path.name[5:]) for path in Path("/dev").glob("video*") if path.name[5:].isdigit()
    )
    return found or list(range(10))


def open_camera(camera_index: str, width: int, height: int):
    attempted = []
    for index in camera_candidates(camera_index):
        attempted.append(index)
        capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            capture.release()
            continue
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(5):
            success, frame = capture.read()
            if success and frame is not None and frame.size:
                return capture, index, frame
        capture.release()
    raise RuntimeError(f"No camera returned a frame; tried indices {attempted}")


def run_self_test(model: Path, threads: int) -> int:
    classifier = Classifier(model, threads)
    frame = np.full((480, 640, 3), 150, np.uint8)
    cv2.circle(frame, (320, 240), 25, (20, 20, 20), -1)
    pixels, region = prepare_frame(frame)
    score, heatmap = classifier.predict(pixels)
    boxes = occlusion_boxes(classifier, pixels, region, score)
    image = draw_preview(frame, region, active=True, now=0.0, ok_score=min(score, 0.49), boxes=boxes)
    assert image.shape == frame.shape and image.dtype == np.uint8
    assert blink_visible(0.0) and not blink_visible(0.25)
    commands: Queue[str] = Queue()
    commands.put("1")
    assert apply_commands(commands, False) == (True, False)
    commands.put("0")
    assert apply_commands(commands, True) == (False, True)
    print(f"Self-test passed: score={score:.4f}, heatmap={heatmap.shape}, boxes={len(boxes)}")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/classify_localize/casting_classify_localize_int8.tflite"))
    parser.add_argument("--camera-index", default="auto")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.camera_index != "auto" and (not args.camera_index.isdigit() or int(args.camera_index) < 0):
        parser.error("camera index must be auto or a nonnegative integer")
    if min(args.width, args.height, args.threads) < 1:
        parser.error("width, height, and threads must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test(args.model, args.threads)
    classifier = Classifier(args.model, args.threads)
    capture, index, frame = open_camera(args.camera_index, args.width, args.height)
    commands: Queue[str] = Queue()
    Thread(target=terminal_reader, args=(commands, sys.stdin), daemon=True).start()
    print(f"Using /dev/video{index}. Type 1 then Enter to start inference; 0 then Enter to stop.", flush=True)
    scores: deque[float] = deque(maxlen=5)
    consecutive_defects = 0
    defect_reported = False
    active = False
    boxes: list[tuple[int, int, int, int, float]] = []
    last_localization = float("-inf")
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        while True:
            active, exit_requested = apply_commands(commands, active)
            if exit_requested:
                break
            now = time.monotonic()
            if active:
                pixels, region = prepare_frame(frame)
                score, _ = classifier.predict(pixels)
                scores.append(score)
                stable_score = float(np.median(scores))
                consecutive_defects = consecutive_defects + 1 if stable_score < 0.5 else 0
                if consecutive_defects >= 3 and not defect_reported:
                    print(f"Found defect (confidence {1 - stable_score:.1%}).", flush=True)
                    defect_reported = True
                elif consecutive_defects == 0:
                    defect_reported = False
                if consecutive_defects >= 3 and now - last_localization >= 0.5:
                    boxes = occlusion_boxes(classifier, pixels, region, score)
                    last_localization = now
                elif consecutive_defects < 3:
                    boxes = []
                    last_localization = float("-inf")
                display = draw_preview(frame, region, active=True, now=now, ok_score=stable_score, boxes=boxes)
            else:
                display = draw_preview(frame, focus_region(frame), active=False, now=now)
            cv2.imshow(WINDOW, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
            success, frame = capture.read()
            if not success or frame is None or frame.size == 0:
                raise RuntimeError(f"Camera /dev/video{index} stopped returning frames")
    finally:
        capture.release()
        cv2.destroyAllWindows()
    print("Camera inference stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
