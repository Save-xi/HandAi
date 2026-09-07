import numpy as np
import pytest

from perception.base import HandDetection
from pipeline import HandPipeline
from utils.config import load_config


def _detection(synthetic_hand_pose, handedness="Right"):
    _, points = synthetic_hand_pose("open")
    xyz = [(0.5 + x * 0.08, 0.2 + y * 0.12, z * 0.08) for x, y, z in points]
    return HandDetection([(x, y) for x, y, _ in xyz], xyz, handedness, 0.99)


def test_device_keypoints_and_injected_detector_produce_the_same_ai_result(synthetic_hand_pose):
    detection = _detection(synthetic_hand_pose)

    class Detector:
        def detect(self, _frame):
            return [detection]

    cfg = load_config("configs/ai.yaml")
    image_pipeline = HandPipeline(cfg, detector=Detector())
    keypoint_pipeline = HandPipeline(cfg)
    for index in range(3):
        first = image_pipeline.process_frame(
            np.zeros((2, 2, 3), np.uint8), frame_index=index, timestamp=10 + index / 30, fps=30
        )
        second = keypoint_pipeline.process_detections([detection], frame_index=index, timestamp=10 + index / 30, fps=30)
        first.pop("latency_ms")
        second.pop("latency_ms")
        assert first == second
    assert second["gesture_stable"] == "open"
    assert second["svh_preview"]["valid"] is True
    assert len(second["svh_preview"]["target_positions"]) == 9


def test_left_hand_and_missing_hand_clear_control(synthetic_hand_pose):
    pipeline = HandPipeline(load_config("configs/ai.yaml"))
    for index, detections in enumerate(([_detection(synthetic_hand_pose, "Left")], [])):
        payload = pipeline.process_detections(detections, frame_index=index, timestamp=10 + index)
        assert payload["detected"] is False
        assert payload["control_ready"] is False
        assert payload["svh_preview"]["valid"] is False


def test_image_entry_requires_a_detector():
    pipeline = HandPipeline(load_config("configs/ai.yaml"))
    with pytest.raises(ValueError, match="detector"):
        pipeline.process_frame(None, frame_index=0)
