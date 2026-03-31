from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from capture.input_source import InputSource


class WebcamSource(InputSource):
    def __init__(self, camera_index: int, width: int, height: int) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.cap.isOpened():
            return False, None
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
