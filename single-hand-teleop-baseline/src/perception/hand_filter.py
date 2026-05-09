from __future__ import annotations

from typing import List, Optional

from perception.mediapipe_hand import HandDetection


def select_right_hand(detections: List[HandDetection]) -> Optional[HandDetection]:
    """从所有检测结果里选出置信度最高的 Right 手。"""
    right_hands = [d for d in detections if isinstance(d.handedness, str) and d.handedness.lower() == "right"]
    if not right_hands:
        return None
    return sorted(right_hands, key=lambda x: x.confidence, reverse=True)[0]
