from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def _fmt_float(value) -> str:
    if value is None:
        return "None"
    return f"{float(value):.3f}"


def build_status_panel(height: int, width: int, data: Dict) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    text_rows = [
        f"detected: {data.get('detected')}",
        f"handedness: {data.get('handedness')}",
        f"gesture: {data.get('gesture')}",
        f"gesture_raw: {data.get('gesture_raw')}",
        f"pinch_distance_norm: {_fmt_float(data.get('pinch_distance_norm'))}",
        f"hand_open_ratio: {_fmt_float(data.get('hand_open_ratio'))}",
        f"thumb curl: {_fmt_float(data.get('finger_curl', {}).get('thumb'))}",
        f"index curl: {_fmt_float(data.get('finger_curl', {}).get('index'))}",
        f"middle curl: {_fmt_float(data.get('finger_curl', {}).get('middle'))}",
        f"ring curl: {_fmt_float(data.get('finger_curl', {}).get('ring'))}",
        f"little curl: {_fmt_float(data.get('finger_curl', {}).get('little'))}",
        f"fps: {data.get('fps', 0.0):.2f}",
        f"latency_ms: {data.get('latency_ms', 0.0):.2f}",
    ]
    y = 30
    for row in text_rows:
        cv2.putText(panel, row, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        y += 28
    return panel
