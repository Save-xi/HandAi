from gesture.rule_based_gesture import GestureStabilizer, infer_gesture_raw, infer_stable_gesture


def test_infer_gesture_raw_categories():
    cfg = {
        "pinch_distance_norm_threshold": 0.45,
        "pinch_open_ratio_min": 0.75,
        "pinch_support_curl_max": 0.65,
        "open_ratio_threshold": 0.85,
        "open_mean_curl_max": 0.45,
        "fist_ratio_threshold": 0.85,
        "fist_mean_curl_min": 0.45,
        "fist_compact_ratio_threshold": 0.65,
    }

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

    assert infer_gesture_raw(open_features, cfg) == "open"
    assert infer_gesture_raw(fist_features, cfg) == "fist"
    assert infer_gesture_raw(pinch_features, cfg) == "pinch"


def test_open_and_fist_rules_do_not_depend_on_thumb_too_much():
    cfg = {
        "pinch_distance_norm_threshold": 0.45,
        "pinch_open_ratio_min": 0.75,
        "pinch_support_curl_max": 0.65,
        "open_ratio_threshold": 0.85,
        "open_mean_curl_max": 0.45,
        "fist_ratio_threshold": 0.85,
        "fist_mean_curl_min": 0.45,
        "fist_compact_ratio_threshold": 0.65,
    }

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

    assert infer_gesture_raw(open_features, cfg) == "open"
    assert infer_gesture_raw(fist_features, cfg) == "fist"


def test_fist_can_use_compact_hand_fallback_when_curl_is_unstable():
    cfg = {
        "pinch_distance_norm_threshold": 0.45,
        "pinch_open_ratio_min": 0.75,
        "pinch_support_curl_max": 0.65,
        "open_ratio_threshold": 0.85,
        "open_mean_curl_max": 0.45,
        "fist_ratio_threshold": 0.85,
        "fist_mean_curl_min": 0.45,
        "fist_compact_ratio_threshold": 0.65,
    }

    fist_features = {
        "detected": True,
        "pinch_distance_norm": 0.32,
        "hand_open_ratio": 0.40,
        "finger_curl": {"thumb": 0.0, "index": 0.0, "middle": 0.0, "ring": 0.0, "little": 0.0},
    }

    assert infer_gesture_raw(fist_features, cfg) == "fist"


def test_infer_stable_gesture_uses_majority_and_unknown_on_missing_hand():
    cfg = {
        "stable_gesture_window": 5,
        "stable_gesture_min_consecutive": 2,
        "stable_unknown_consecutive": 1,
    }
    history = [
        {"detected": True, "gesture_raw": "pinch"},
        {"detected": True, "gesture_raw": "pinch"},
        {"detected": True, "gesture_raw": "fist"},
        {"detected": True, "gesture_raw": "fist"},
    ]

    assert infer_stable_gesture(history, cfg) == "fist"
    assert infer_stable_gesture(history + [{"detected": False, "gesture_raw": "unknown"}], cfg) == "unknown"


def test_gesture_stabilizer_switches_without_old_history_blocking_new_gesture():
    stabilizer = GestureStabilizer(confirm_frames=2, unknown_confirm_frames=1)

    assert stabilizer.update("pinch") == "unknown"
    assert stabilizer.update("pinch") == "pinch"
    assert stabilizer.update("fist") == "pinch"
    assert stabilizer.update("fist") == "fist"
    assert stabilizer.update("unknown") == "unknown"
