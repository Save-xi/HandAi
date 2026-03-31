from __future__ import annotations

import cv2
import numpy as np

from visualize.status_panel import build_status_panel


def compose_view(left_bgr: np.ndarray, status_data: dict) -> np.ndarray:
    h, w, _ = left_bgr.shape
    panel = build_status_panel(h, max(380, w // 3), status_data)
    return cv2.hconcat([left_bgr, panel])
