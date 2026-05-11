from __future__ import annotations

"""摄像头输入源。"""

from typing import Optional, Tuple

import cv2
import numpy as np

from capture.input_source import InputSource


class WebcamSource(InputSource):
    """用 OpenCV VideoCapture 读取实时摄像头画面。"""

    def __init__(self, camera_index: int, width: int, height: int) -> None:
        # 这里设置的是期望分辨率，实际能否生效取决于摄像头驱动。
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

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
