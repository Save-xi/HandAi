from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List


class RecentFrameBuffer:
    """Small in-memory frame history for debounce and future temporal models."""

    def __init__(self, maxlen: int = 10) -> None:
        self.frames: Deque[Dict] = deque(maxlen=maxlen)

    def append(self, payload: Dict) -> None:
        self.frames.append(
            {
                "timestamp": payload.get("timestamp"),
                "detected": payload.get("detected"),
                "gesture_raw": payload.get("gesture_raw"),
                "pinch_distance_norm": payload.get("pinch_distance_norm"),
                "hand_open_ratio": payload.get("hand_open_ratio"),
                "finger_curl": dict(payload.get("finger_curl", {})),
                "landmarks_2d": list(payload.get("landmarks_2d", [])),
            }
        )

    def as_list(self) -> List[Dict]:
        return list(self.frames)
