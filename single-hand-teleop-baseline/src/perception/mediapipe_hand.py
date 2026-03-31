from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp


@dataclass
class HandDetection:
    landmarks_2d: List[Tuple[float, float]]
    handedness: str
    confidence: float


class MediaPipeHandDetector:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "hands"):
            raise RuntimeError(
                "Unsupported mediapipe package detected. "
                "Install mediapipe==0.10.14 for this baseline."
            )
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, bgr_frame) -> List[HandDetection]:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        detections: List[HandDetection] = []
        if not results.multi_hand_landmarks or not results.multi_handedness:
            return detections

        for lm, handness in zip(results.multi_hand_landmarks, results.multi_handedness):
            label = handness.classification[0].label
            score = float(handness.classification[0].score)
            points = [(float(p.x), float(p.y)) for p in lm.landmark]
            detections.append(HandDetection(landmarks_2d=points, handedness=label, confidence=score))
        return detections

    def draw_landmarks(self, frame, landmarks_2d: List[Tuple[float, float]]) -> None:
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
