from __future__ import annotations

"""感知模型的公共输入输出；设备采集与具体模型由各自适配器实现。"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class HandDetection:
    """单手 21 点，按 MediaPipe 关节顺序排列。

    xy 为图像归一化坐标；xyz 保留 MediaPipe 原始 x/y/z，不能当作毫米。
    x、y 分别按图像宽、高归一化，z 约为图像宽尺度。M3-A 必须提供
    image_width/image_height；流水线另算 (x, y*H/W, z) 用于几何。
    handedness 必须是按真实左右手修正后的 Right/Left。
    """

    landmarks_2d: list[tuple[float, float]]
    landmarks_xyz: list[tuple[float, float, float]]
    handedness: str
    confidence: float
    image_width: int | None = None
    image_height: int | None = None
    coordinate_space: str | None = None


class HandDetector(Protocol):
    """MediaPipe、MMPose 或自定义姿态模型的接入入口。"""

    def detect(self, bgr_frame: np.ndarray) -> list[HandDetection]: ...

    def close(self) -> None: ...
