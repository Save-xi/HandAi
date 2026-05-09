from __future__ import annotations

from typing import Dict, Iterable, List


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def _tail_count(values: List[str], target: str) -> int:
    count = 0
    for value in reversed(values):
        if value != target:
            break
        count += 1
    return count


def infer_gesture_raw(features: Dict, cfg: Dict) -> str:
    if not features.get("detected", False):
        return "unknown"

    pinch_threshold = float(cfg.get("pinch_distance_norm_threshold", 0.45))
    pinch_open_ratio_min = float(cfg.get("pinch_open_ratio_min", 0.75))
    pinch_support_curl_max = float(cfg.get("pinch_support_curl_max", 0.65))
    open_ratio_threshold = float(cfg.get("open_ratio_threshold", 0.85))
    open_mean_curl_max = float(cfg.get("open_mean_curl_max", 0.45))
    fist_ratio_threshold = float(cfg.get("fist_ratio_threshold", 0.85))
    fist_mean_curl_min = float(cfg.get("fist_mean_curl_min", 0.45))
    fist_compact_ratio_threshold = float(cfg.get("fist_compact_ratio_threshold", 0.65))

    pinch_distance = features.get("pinch_distance_norm")
    open_ratio = features.get("hand_open_ratio")
    curls = features["finger_curl"]
    if pinch_distance is None or open_ratio is None:
        return "unknown"

    support_curls = [curls["middle"], curls["ring"], curls["little"]]
    non_thumb_curls = [curls["index"], curls["middle"], curls["ring"], curls["little"]]

    # 先判断 fist，因为紧凑手型在 2D 里往往也会表现出较短的拇指-食指距离，
    # 否则容易被误判成 pinch。
    if float(open_ratio) <= fist_compact_ratio_threshold:
        return "fist"

    # Pinch 要求拇指与食指足够接近，同时整只手不能太紧凑。
    if (
        float(pinch_distance) <= pinch_threshold
        and float(open_ratio) >= pinch_open_ratio_min
        and _mean(support_curls) <= pinch_support_curl_max
    ):
        return "pinch"

    # Open 主要由四个非拇指手指决定；
    # 拇指姿态在人和视角之间变化很大，不应该主导这条规则。
    if float(open_ratio) >= open_ratio_threshold and _mean(non_thumb_curls) <= open_mean_curl_max:
        return "open"

    # 当 ratio 处在边界附近时，再用 curl 辅助判定 fist。
    if float(open_ratio) <= fist_ratio_threshold and _mean(non_thumb_curls) >= fist_mean_curl_min:
        return "fist"

    return "unknown"


def infer_gesture(features: Dict, cfg: Dict) -> str:
    return infer_gesture_raw(features, cfg)


class GestureStabilizer:
    """只依赖连续命中的、与顺序无关的手势去抖器。"""

    def __init__(self, confirm_frames: int = 2, unknown_confirm_frames: int = 1) -> None:
        self.confirm_frames = max(1, confirm_frames)
        self.unknown_confirm_frames = max(1, unknown_confirm_frames)
        self.stable_gesture = "unknown"
        self.candidate_gesture: str | None = None
        self.candidate_count = 0

    def update(self, raw_gesture: str | None) -> str:
        raw = raw_gesture or "unknown"

        if raw == self.stable_gesture:
            self.candidate_gesture = None
            self.candidate_count = 0
            return self.stable_gesture

        if raw == self.candidate_gesture:
            self.candidate_count += 1
        else:
            self.candidate_gesture = raw
            self.candidate_count = 1

        threshold = self.unknown_confirm_frames if raw == "unknown" else self.confirm_frames
        if self.candidate_count >= threshold:
            self.stable_gesture = raw
            self.candidate_gesture = None
            self.candidate_count = 0

        return self.stable_gesture


def infer_stable_gesture(history: List[Dict], cfg: Dict) -> str:
    if not history:
        return "unknown"

    window_size = int(cfg.get("stable_gesture_window", 5))
    min_consecutive = int(cfg.get("stable_gesture_min_consecutive", 2))
    unknown_consecutive = int(cfg.get("stable_unknown_consecutive", 1))

    window = history[-window_size:]
    raw_gestures = [(frame.get("gesture_raw") or "unknown") for frame in window]
    latest_raw = raw_gestures[-1]

    threshold = unknown_consecutive if latest_raw == "unknown" else min_consecutive
    if _tail_count(raw_gestures, latest_raw) >= threshold:
        return latest_raw

    return "unknown"
