from __future__ import annotations

"""最近帧摘要缓存。"""

from collections import deque
from typing import Deque, Dict, List


class RecentFrameBuffer:
    """用于去抖、调试和未来时序模型的小型内存帧历史。

    这里不保存完整 payload，只保存常用摘要字段。这样既减少内存占用，
    也避免把大段 landmark / preview 数组在内存里重复堆积。
    """

    def __init__(self, maxlen: int = 10) -> None:
        self.frames: Deque[Dict] = deque(maxlen=maxlen)

    def append(self, payload: Dict) -> None:
        """追加一帧摘要。"""

        self.frames.append(
            {
                "timestamp": payload.get("timestamp"),
                "frame_index": payload.get("frame_index"),
                "detected": payload.get("detected"),
                "control_ready": payload.get("control_ready"),
                "gesture_raw": payload.get("gesture_raw"),
                "gesture_stable": payload.get("gesture_stable"),
                "pinch_distance_norm": payload.get("pinch_distance_norm"),
                "hand_open_ratio": payload.get("hand_open_ratio"),
                "finger_curl": dict(payload.get("finger_curl", {})),
                "landmarks_2d": list(payload.get("landmarks_2d", [])),
            }
        )

    def as_list(self) -> List[Dict]:
        return list(self.frames)
