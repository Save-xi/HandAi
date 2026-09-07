from __future__ import annotations

"""感知模型的公共输入输出；设备采集与具体模型由各自适配器实现。"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class HandDetection:
    """单手 21 点，按 MediaPipe 关节顺序排列。

    xy 为图像归一化坐标；xyz 为同尺度相对坐标，不能直接当作毫米。
    handedness 必须是按真实左右手修正后的 Right/Left。
    """

    landmarks_2d: list[tuple[float, float]]
    landmarks_xyz: list[tuple[float, float, float]]
    handedness: str
    confidence: float


class HandDetector(Protocol):
    """MediaPipe、MMPose 或自定义姿态模型的接入入口。"""

    def detect(self, bgr_frame: np.ndarray) -> list[HandDetection]: ...

    def close(self) -> None: ...
