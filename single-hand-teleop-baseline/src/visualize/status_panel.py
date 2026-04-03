from __future__ import annotations

from typing import Dict

import cv2
import numpy as np
from output.frame_payload_contract import get_stable_gesture, get_svh_preview


def _fmt_float(value) -> str:
    if value is None:
        return "None"
    return f"{float(value):.3f}"


def _fmt_list_preview(values, max_items: int = 3) -> str:
    if not values:
        return "[]"
    preview = ", ".join(f"{float(v):.2f}" for v in values[:max_items])
    if len(values) > max_items:
        preview += ", ..."
    return f"[{preview}]"


def build_status_panel(height: int, width: int, data: Dict) -> np.ndarray:
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    svh = get_svh_preview(data)
    control = data.get("control_representation", {})
    text_rows = [
        f"detected: {data.get('detected')}",
        f"handedness: {data.get('handedness')}",
        f"gesture_stable: {get_stable_gesture(data)}",
        f"gesture_raw: {data.get('gesture_raw')}",
        f"control_ready: {data.get('control_ready')}",
        f"pinch_distance_norm: {_fmt_float(data.get('pinch_distance_norm'))}",
        f"hand_open_ratio: {_fmt_float(data.get('hand_open_ratio'))}",
        f"ctrl.features: {control.get('features_valid')}",
        f"grasp_close: {_fmt_float(control.get('grasp_close'))}",
        f"pinch_eff: {_fmt_float(control.get('effective_pinch_strength'))}",
        f"thumb_index_prox: {_fmt_float(control.get('thumb_index_proximity'))}",
        f"thumb curl: {_fmt_float(data.get('finger_curl', {}).get('thumb'))}",
        f"index curl: {_fmt_float(data.get('finger_curl', {}).get('index'))}",
        f"middle curl: {_fmt_float(data.get('finger_curl', {}).get('middle'))}",
        f"ring curl: {_fmt_float(data.get('finger_curl', {}).get('ring'))}",
        f"little curl: {_fmt_float(data.get('finger_curl', {}).get('little'))}",
        f"svh_preview.valid: {svh.get('valid')}",
        f"svh_preview.mode: {svh.get('mode')}",
        f"svh_preview.targets: {_fmt_list_preview(svh.get('target_positions', []))}",
        f"fps: {data.get('fps', 0.0):.2f}",
        f"latency_ms: {data.get('latency_ms', 0.0):.2f}",
    ]
    y = 30
    for row in text_rows:
        cv2.putText(panel, row, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        y += 28
    return panel
