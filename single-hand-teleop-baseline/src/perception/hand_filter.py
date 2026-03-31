from __future__ import annotations

from typing import List, Optional

from perception.mediapipe_hand import HandDetection


def select_right_hand(detections: List[HandDetection]) -> Optional[HandDetection]:
    """Pick the most confident Right hand from all detections."""
    right_hands = [d for d in detections if d.handedness.lower() == "right"]
    if not right_hands:
        return None
    return sorted(right_hands, key=lambda x: x.confidence, reverse=True)[0]
