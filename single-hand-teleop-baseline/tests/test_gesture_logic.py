import pytest

from gesture.rule_based_gesture import GestureStabilizer, infer_gesture_raw, infer_stable_gesture


def test_infer_gesture_raw_categories(gesture_cfg):
    open_features = {
        "detected": True,
        "pinch_distance_norm": 0.8,
        "hand_open_ratio": 2.0,
        "finger_curl": {"thumb": 0.1, "index": 0.1, "middle": 0.1, "ring": 0.1, "little": 0.1},
    }
    fist_features = {
        "detected": True,
        "pinch_distance_norm": 0.9,
        "hand_open_ratio": 0.8,
        "finger_curl": {"thumb": 0.8, "index": 0.8, "middle": 0.8, "ring": 0.8, "little": 0.8},
    }
    pinch_features = {
        "detected": True,
        "pinch_distance_norm": 0.3,
        "hand_open_ratio": 1.4,
        "finger_curl": {"thumb": 0.4, "index": 0.4, "middle": 0.2, "ring": 0.2, "little": 0.2},
    }

    assert infer_gesture_raw(open_features, gesture_cfg) == "open"
    assert infer_gesture_raw(fist_features, gesture_cfg) == "fist"
    assert infer_gesture_raw(pinch_features, gesture_cfg) == "pinch"


def test_open_and_fist_rules_do_not_depend_on_thumb_too_much(gesture_cfg):
    open_features = {
        "detected": True,
        "pinch_distance_norm": 0.9,
        "hand_open_ratio": 1.3,
        "finger_curl": {"thumb": 0.75, "index": 0.1, "middle": 0.1, "ring": 0.1, "little": 0.1},
    }
    fist_features = {
        "detected": True,
        "pinch_distance_norm": 0.9,
        "hand_open_ratio": 0.7,
        "finger_curl": {"thumb": 0.10, "index": 0.7, "middle": 0.7, "ring": 0.7, "little": 0.7},
    }

    assert infer_gesture_raw(open_features, gesture_cfg) == "open"
    assert infer_gesture_raw(fist_features, gesture_cfg) == "fist"


def test_fist_can_use_compact_hand_fallback_when_curl_is_unstable(gesture_cfg):
    fist_features = {
        "detected": True,
        "pinch_distance_norm": 0.32,
        "hand_open_ratio": 0.40,
        "finger_curl": {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "little": 0.0},
    }

    assert infer_gesture_raw(fist_features, gesture_cfg) == "fist"


def test_infer_gesture_raw_returns_unknown_when_control_features_are_cleared(gesture_cfg):
    degraded_features = {
        "detected": True,
        "pinch_distance_norm": None,
        "hand_open_ratio": None,
        "finger_curl": {"thumb": None, "index": None, "middle": None, "ring": None, "little": None},
    }

    assert infer_gesture_raw(degraded_features, gesture_cfg) == "unknown"


@pytest.mark.parametrize(
    ("features", "expected"),
    [
        (
            {
                "detected": True,
                "pinch_distance_norm": 0.45,
                "hand_open_ratio": 0.75,
                "finger_curl": {"thumb": 0.4, "index": 0.35, "middle": 0.65, "ring": 0.65, "little": 0.65},
            },
            "pinch",
        ),
        (
            {
                "detected": True,
                "pinch_distance_norm": 0.44,
                "hand_open_ratio": 0.86,
                "finger_curl": {"thumb": 0.4, "index": 0.35, "middle": 0.66, "ring": 0.66, "little": 0.66},
            },
            "unknown",
        ),
        (
            {
                "detected": True,
                "pinch_distance_norm": 0.90,
                "hand_open_ratio": 0.85,
                "finger_curl": {"thumb": 0.9, "index": 0.45, "middle": 0.45, "ring": 0.45, "little": 0.45},
            },
            "open",
        ),
        (
            {
                "detected": True,
                "pinch_distance_norm": 0.90,
                "hand_open_ratio": 0.65,
                "finger_curl": {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "little": 0.0},
            },
            "fist",
        ),
        (
            {
                "detected": True,
                "pinch_distance_norm": 0.90,
                "hand_open_ratio": 0.85,
                "finger_curl": {"thumb": 0.0, "index": 0.45, "middle": 0.45, "ring": 0.45, "little": 0.45},
            },
            "open",
        ),
    ],
)
def test_infer_gesture_raw_boundary_thresholds(gesture_cfg, features, expected):
    assert infer_gesture_raw(features, gesture_cfg) == expected


def test_infer_stable_gesture_uses_majority_and_unknown_on_missing_hand(gesture_cfg):
    history = [
        {"detected": True, "gesture_raw": "pinch"},
        {"detected": True, "gesture_raw": "pinch"},
        {"detected": True, "gesture_raw": "fist"},
        {"detected": True, "gesture_raw": "fist"},
    ]

    assert infer_stable_gesture(history, gesture_cfg) == "fist"
    assert infer_stable_gesture(history + [{"detected": False, "gesture_raw": "unknown"}], gesture_cfg) == "unknown"


def test_gesture_stabilizer_switches_without_old_history_blocking_new_gesture():
    stabilizer = GestureStabilizer(confirm_frames=2, unknown_confirm_frames=1)

    assert stabilizer.update("pinch") == "unknown"
    assert stabilizer.update("pinch") == "pinch"
    assert stabilizer.update("fist") == "pinch"
    assert stabilizer.update("fist") == "fist"
    assert stabilizer.update("unknown") == "unknown"


def test_gesture_stabilizer_requires_confirm_frames_for_known_labels_but_not_unknown():
    stabilizer = GestureStabilizer(confirm_frames=3, unknown_confirm_frames=1)

    assert stabilizer.update("open") == "unknown"
    assert stabilizer.update("open") == "unknown"
    assert stabilizer.update("open") == "open"
    assert stabilizer.update("fist") == "open"
    assert stabilizer.update("unknown") == "unknown"
