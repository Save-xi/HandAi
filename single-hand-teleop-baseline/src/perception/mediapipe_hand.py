from __future__ import annotations

"""MediaPipe 手部检测封装。

本模块只负责“从图像里拿到手部关键点和左右手标签”。
右手筛选、特征提取、手势识别都放在后续模块里，避免检测器承担过多职责。
"""

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp


@dataclass
class HandDetection:
    """单只手的检测结果。

    landmarks_2d:
        MediaPipe 输出的归一化图像坐标，范围通常接近 [0, 1]。
    landmarks_xyz:
        MediaPipe 输出的 3D-like 坐标。这里主要用于增强 curl 估计，
        不能理解成真实世界毫米坐标。
    handedness:
        已根据 input_mirrored 修正后的左右手标签。
    confidence:
        MediaPipe 对左右手分类的置信度。
    """

    landmarks_2d: List[Tuple[float, float]]
    landmarks_xyz: List[Tuple[float, float, float]]
    handedness: str
    confidence: float


def normalize_handedness(label: str, input_mirrored: bool) -> str:
    """修正 MediaPipe 的左右手标签。

    MediaPipe Hands 默认按自拍镜像视角解释左右手。普通摄像头画面没有镜像时，
    标签需要翻转，否则用户伸右手可能会被当成 Left。
    """

    if input_mirrored:
        return label
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


class MediaPipeHandDetector:
    """对 MediaPipe Hands 的薄封装。

    这里保留一个类主要是为了集中管理 MediaPipe 对象生命周期：
    创建时初始化 graph，退出时 close，避免主循环直接依赖第三方 API 细节。
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        input_mirrored: bool = False,
        static_image_mode: bool = False,
    ) -> None:
        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
            raise RuntimeError(
                "检测到不受支持的 mediapipe 包。"
                "这个 baseline 请安装 mediapipe==0.10.14。"
            )
        self.input_mirrored = input_mirrored
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, bgr_frame) -> List[HandDetection]:
        """检测一帧 BGR 图像中的所有手，并返回统一结构。"""

        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        detections: List[HandDetection] = []
        if not results.multi_hand_landmarks or not results.multi_handedness:
            return detections

        for lm, handness in zip(results.multi_hand_landmarks, results.multi_handedness):
            raw_label = handness.classification[0].label
            label = normalize_handedness(raw_label, self.input_mirrored)
            score = float(handness.classification[0].score)
            # p.x / p.y 是归一化图像坐标；p.z 是 MediaPipe 的相对深度线索。
            points_xyz = [(float(p.x), float(p.y), float(p.z)) for p in lm.landmark]
            points_2d = [(x, y) for x, y, _ in points_xyz]
            detections.append(
                HandDetection(
                    landmarks_2d=points_2d,
                    landmarks_xyz=points_xyz,
                    handedness=label,
                    confidence=score,
                )
            )
        return detections

    def draw_landmarks(self, frame, landmarks_2d: List[Tuple[float, float]]) -> None:
        """在原图上画出当前选中手的关键点骨架，仅用于 GUI 预览。"""

        h, w, _ = frame.shape
        pixel = [(int(x * w), int(y * h)) for x, y in landmarks_2d]
        for c in self.mp_hands.HAND_CONNECTIONS:
            p1 = pixel[c[0]]
            p2 = pixel[c[1]]
            cv2.line(frame, p1, p2, (0, 255, 0), 2)
        for p in pixel:
            cv2.circle(frame, p, 3, (0, 0, 255), -1)

    def close(self) -> None:
        self.hands.close()
