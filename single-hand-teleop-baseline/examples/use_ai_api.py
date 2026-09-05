from __future__ import annotations

"""协作方关键点回调示例；可直接运行，示例输出一帧无手结果。"""

import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from perception.base import HandDetection  # noqa: E402
from pipeline import HandPipeline  # noqa: E402
from utils.config import load_config  # noqa: E402


def build_device_callback(publish):
    """设备成员提供关键点，下游成员提供 publish(payload) 函数。"""
    pipeline = HandPipeline(load_config(str(PROJECT_ROOT / "configs/ai.yaml")))

    def on_landmarks(detections: list[HandDetection], frame_index: int, timestamp: float):
        payload = pipeline.process_detections(
            detections,
            frame_index=frame_index,
            timestamp=timestamp,
        )
        publish(payload)
        return payload

    return on_landmarks


if __name__ == "__main__":
    callback = build_device_callback(lambda payload: print(json.dumps(payload, ensure_ascii=False)))
    callback([], frame_index=0, timestamp=time.time())
