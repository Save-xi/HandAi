from __future__ import annotations

from typing import Dict

import cv2
import numpy as np


def build_status_panel(height: int, width: int, data: Dict) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    text_rows = [
        f"detected: {data.get('detected')}",
        f"handedness: {data.get('handedness')}",
        f"gesture: {data.get('gesture')}",
        f"pinch_distance: {data.get('pinch_distance', 0.0):.3f}",
        f"hand_open_ratio: {data.get('hand_open_ratio', 0.0):.3f}",
        f"thumb curl: {data.get('finger_curl', {}).get('thumb', 0.0):.3f}",
        f"index curl: {data.get('finger_curl', {}).get('index', 0.0):.3f}",
        f"middle curl: {data.get('finger_curl', {}).get('middle', 0.0):.3f}",
        f"ring curl: {data.get('finger_curl', {}).get('ring', 0.0):.3f}",
        f"little curl: {data.get('finger_curl', {}).get('little', 0.0):.3f}",
        f"fps: {data.get('fps', 0.0):.2f}",
        f"latency_ms: {data.get('latency_ms', 0.0):.2f}",
    ]
    y = 30
    for row in text_rows:
        cv2.putText(panel, row, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        y += 28
    return panel
