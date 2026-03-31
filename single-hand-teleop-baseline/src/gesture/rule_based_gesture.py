from __future__ import annotations

from typing import Dict


def infer_gesture(features: Dict, cfg: Dict) -> str:
    if not features.get("detected", False):
        return "unknown"

    pinch_threshold = float(cfg.get("pinch_threshold", 0.12))
    open_ratio_threshold = float(cfg.get("open_ratio_threshold", 0.55))
    fist_ratio_threshold = float(cfg.get("fist_ratio_threshold", 0.32))
    curl_cfg = cfg.get("finger_curl_thresholds", {})

    pinch_distance = float(features["pinch_distance"])
    open_ratio = float(features["hand_open_ratio"])
    curls = features["finger_curl"]

    # Rule 1: pinch requires thumb-index close and index not fully curled.
    if pinch_distance < pinch_threshold and curls["index"] < 0.85:
        return "pinch"

    # Rule 2: open requires large spread and most fingers not curled.
    if open_ratio > open_ratio_threshold and all(
        curls[k] < float(curl_cfg.get(k, 0.62)) for k in ["index", "middle", "ring", "little"]
    ):
        return "open"

    # Rule 3: fist requires small spread and all fingers curled.
    if open_ratio < fist_ratio_threshold and all(
        curls[k] > float(curl_cfg.get(k, 0.62)) for k in ["thumb", "index", "middle", "ring", "little"]
    ):
        return "fist"

    return "unknown"
