from __future__ import annotations

from typing import List, Optional
import math

from perception.base import HandDetection


def select_right_hand(detections: List[HandDetection]) -> Optional[HandDetection]:
    """从所有检测结果里选出置信度最高的 Right 手。"""
    right_hands = [d for d in detections if isinstance(d.handedness, str) and d.handedness.lower() == "right"]
    if not right_hands:
        return None
    def score(detection):
        try:
            value = float(detection.confidence)
            return value if math.isfinite(value) else -math.inf
        except (ValueError, TypeError, OverflowError):
            return -math.inf

    return max(right_hands, key=score)
