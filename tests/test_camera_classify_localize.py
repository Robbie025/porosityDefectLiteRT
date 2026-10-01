from __future__ import annotations

from queue import Queue

import numpy as np

import camera_classify_localize as camera


def test_focus_crop_and_evidence_mapping() -> None:
    frame = np.zeros((400, 600, 3), np.uint8)
    pixels, region = camera.prepare_frame(frame)
    assert pixels.shape == (300, 300) and pixels.dtype == np.uint8
    assert region == camera.Region(100, 0, 400)
    heatmap = np.zeros((38, 38), np.float32)
    heatmap[10:18, 12:20] = 0.95
    boxes = camera.heatmap_boxes(heatmap, region)
    assert boxes
    assert all(region.x <= x1 <= x2 < region.x + region.side for x1, _, x2, _, _ in boxes)


def test_terminal_state_and_blink() -> None:
    commands: Queue[str] = Queue()
    commands.put("1")
    assert camera.apply_commands(commands, False) == (True, False)
    assert camera.blink_visible(0.0)
    assert not camera.blink_visible(0.25)
    commands.put("0")
    assert camera.apply_commands(commands, True) == (False, True)
    frame = np.zeros((200, 200, 3), np.uint8)
    region = camera.focus_region(frame)
    visible = camera.draw_preview(frame, region, active=True, now=0.0)
    hidden = camera.draw_preview(frame, region, active=True, now=0.25)
    assert not np.array_equal(visible, hidden)


def test_defect_preview_flashes_red_and_keeps_status_visible() -> None:
    frame = np.full((200, 300, 3), 180, np.uint8)
    region = camera.focus_region(frame)
    flashing = camera.draw_preview(frame, region, active=True, now=0.0, ok_score=0.2)
    resting = camera.draw_preview(frame, region, active=True, now=0.25, ok_score=0.2)
    okay = camera.draw_preview(frame, region, active=True, now=0.0, ok_score=0.8)
    assert tuple(flashing[100, 1]) == camera.ALERT_RED
    assert tuple(resting[100, 1]) != camera.ALERT_RED
    assert tuple(flashing[5, 100]) == camera.ALERT_RED
    assert tuple(resting[5, 100]) == (25, 25, 25)
    assert tuple(okay[5, 100]) == (25, 25, 25)


def test_occlusion_localizes_influential_patch() -> None:
    class FakeClassifier:
        def predict(self, pixels):
            # Blurring the upper-left patch removes its checkerboard pattern.
            variation = float(np.std(pixels[0:75, 0:75]))
            return (0.1 if variation > 50 else 0.8), np.zeros((18, 18), np.float32)

    pixels = np.full((300, 300), 128, np.uint8)
    yy, xx = np.indices((75, 75))
    pixels[:75, :75] = ((xx + yy) % 2 * 255).astype(np.uint8)
    boxes = camera.occlusion_boxes(FakeClassifier(), pixels, camera.Region(100, 0, 300), 0.1)
    assert boxes
    assert boxes[0][0] == 100 and boxes[0][1] == 0
