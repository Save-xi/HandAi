from __future__ import annotations

"""视频文件输入源。

用于离线复现实验、跑固定 demo 视频，或在没有摄像头时做 pipeline 验证。
"""

from typing import Optional, Tuple

import cv2
import numpy as np

from capture.input_source import InputSource


class VideoFileSource(InputSource):
    """逐帧读取本地视频文件。"""

    def __init__(self, video_path: str) -> None:
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

    def is_opened(self) -> bool:
        return bool(self.cap is not None and self.cap.isOpened())

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_opened():
            return False, None
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
