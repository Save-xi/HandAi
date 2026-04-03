from perception.hand_filter import select_right_hand
from perception.mediapipe_hand import HandDetection


def _detection(handedness, confidence):
    return HandDetection(
        landmarks_2d=[(0.0, 0.0)] * 21,
        landmarks_xyz=[(0.0, 0.0, 0.0)] * 21,
        handedness=handedness,
        confidence=confidence,
    )


def test_select_right_hand_prefers_highest_confidence_right_hand():
    chosen = select_right_hand(
        [
            _detection("Left", 0.99),
            _detection("Right", 0.61),
            _detection("RIGHT", 0.93),
            _detection("Right", 0.72),
        ]
    )

    assert chosen is not None
    assert chosen.handedness == "RIGHT"
    assert chosen.confidence == 0.93


def test_select_right_hand_returns_none_when_no_right_hand_exists():
    assert select_right_hand([_detection("Left", 0.9), _detection("left", 0.8)]) is None


def test_select_right_hand_ignores_malformed_handedness_entries():
    chosen = select_right_hand(
        [
            _detection(None, 0.99),
            _detection("Right", 0.62),
        ]
    )

    assert chosen is not None
    assert chosen.handedness == "Right"
