from perception.mediapipe_hand import MediaPipeHandDetector, normalize_handedness


def test_mediapipe_detector_init_close():
    detector = MediaPipeHandDetector(max_num_hands=1)
    detector.close()


def test_normalize_handedness_for_non_mirrored_input():
    assert normalize_handedness("Left", input_mirrored=False) == "Right"
    assert normalize_handedness("Right", input_mirrored=False) == "Left"


def test_normalize_handedness_for_mirrored_input():
    assert normalize_handedness("Left", input_mirrored=True) == "Left"
    assert normalize_handedness("Right", input_mirrored=True) == "Right"
