from perception.mediapipe_hand import MediaPipeHandDetector, normalize_handedness
from types import SimpleNamespace
import numpy as np


def test_mediapipe_detector_init_close():
    detector = MediaPipeHandDetector(max_num_hands=1)
    detector.close()


def test_normalize_handedness_for_non_mirrored_input():
    assert normalize_handedness("Left", input_mirrored=False) == "Right"
    assert normalize_handedness("Right", input_mirrored=False) == "Left"


def test_normalize_handedness_for_mirrored_input():
    assert normalize_handedness("Left", input_mirrored=True) == "Left"
    assert normalize_handedness("Right", input_mirrored=True) == "Right"


def test_normalize_handedness_leaves_unknown_labels_unchanged():
    assert normalize_handedness("Unknown", input_mirrored=False) == "Unknown"


def test_detection_declares_original_image_dimensions_and_coordinate_domain():
    detector = MediaPipeHandDetector.__new__(MediaPipeHandDetector)
    detector.input_mirrored = False
    result = SimpleNamespace(
        multi_hand_landmarks=[SimpleNamespace(landmark=[SimpleNamespace(x=0.4, y=0.6, z=-0.1)] * 21)],
        multi_handedness=[SimpleNamespace(classification=[SimpleNamespace(label="Left", score=0.9)])],
    )
    detector.hands = SimpleNamespace(process=lambda _rgb: result)
    detection = detector.detect(np.zeros((1080, 1920, 3), dtype=np.uint8))[0]
    assert (detection.image_width, detection.image_height) == (1920, 1080)
    assert detection.coordinate_space == "mediapipe_image_xyz" and detection.handedness == "Right"
    assert detection.landmarks_xyz[0] == (0.4, 0.6, -0.1)
