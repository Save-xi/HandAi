from control.control_representation import build_control_representation


def _control_cfg():
    return {
        "control_grasp_open_ref": 0.02,
        "control_grasp_closed_ref": 0.55,
        "control_pinch_open_ref": 0.45,
        "control_pinch_closed_ref": 0.08,
        "control_hand_open_ratio_open_ref": 0.95,
        "control_hand_open_ratio_closed_ref": 0.25,
        "control_pinch_index_open_ref": 0.05,
        "control_pinch_index_closed_ref": 0.35,
    }


def _payload(gesture: str, hand_open_ratio, pinch_distance_norm, finger_curl, detected: bool = True):
    return {
        "detected": detected,
        "gesture_raw": gesture,
        "gesture_stable": gesture,
        "hand_open_ratio": hand_open_ratio,
        "pinch_distance_norm": pinch_distance_norm,
        "finger_curl": finger_curl,
    }


def test_control_representation_invalid_when_control_features_missing():
    control = build_control_representation(
        _payload(
            "unknown",
            None,
            None,
            {"thumb": None, "index": None, "middle": None, "ring": None, "little": None},
            detected=False,
        ),
        _control_cfg(),
    )

    assert control["valid"] is False
    assert control["features_valid"] is False
    assert control["command_ready"] is False
    assert control["grasp_close"] is None
    assert control["thumb_index_proximity"] is None
    assert control["effective_pinch_strength"] is None
    assert control["pinch_strength"] is None


def test_control_representation_separates_open_from_fist():
    cfg = _control_cfg()
    open_control = build_control_representation(
        _payload(
            "open",
            0.95,
            0.82,
            {"thumb": 0.03, "index": 0.02, "middle": 0.02, "ring": 0.02, "little": 0.02},
        ),
        cfg,
    )
    fist_control = build_control_representation(
        _payload(
            "fist",
            0.24,
            0.28,
            {"thumb": 0.14, "index": 0.53, "middle": 0.57, "ring": 0.51, "little": 0.44},
        ),
        cfg,
    )

    assert open_control["features_valid"] is True
    assert fist_control["features_valid"] is True
    assert open_control["command_ready"] is True
    assert fist_control["command_ready"] is True
    assert open_control["preferred_mapping"] == "grasp"
    assert fist_control["preferred_mapping"] == "grasp"
    assert open_control["grasp_close"] < fist_control["grasp_close"]
    assert open_control["effective_pinch_strength"] == 0.0
    assert fist_control["effective_pinch_strength"] == 0.0


def test_control_representation_pinch_strength_rises_for_pinch_pose():
    cfg = _control_cfg()
    pinch_control = build_control_representation(
        _payload(
            "pinch",
            0.83,
            0.12,
            {"thumb": 0.04, "index": 0.18, "middle": 0.02, "ring": 0.02, "little": 0.02},
        ),
        cfg,
    )

    assert pinch_control["features_valid"] is True
    assert pinch_control["command_ready"] is True
    assert pinch_control["preferred_mapping"] == "pinch"
    assert pinch_control["thumb_index_proximity"] > 0.5
    assert pinch_control["effective_pinch_strength"] > 0.5
    assert pinch_control["pinch_strength"] == pinch_control["effective_pinch_strength"]
    assert pinch_control["finger_flex"]["index"] > pinch_control["finger_flex"]["middle"]
    assert pinch_control["valid"] == pinch_control["command_ready"]


def test_control_representation_clamps_continuous_outputs_into_unit_interval():
    control = build_control_representation(
        _payload(
            "pinch",
            10.0,
            -3.0,
            {"thumb": 2.5, "index": -1.0, "middle": 1.8, "ring": 0.4, "little": 0.2},
        ),
        _control_cfg(),
    )

    assert 0.0 <= control["grasp_close"] <= 1.0
    assert 0.0 <= control["thumb_index_proximity"] <= 1.0
    assert 0.0 <= control["effective_pinch_strength"] <= 1.0
    assert 0.0 <= control["support_flex"] <= 1.0
    assert all(0.0 <= value <= 1.0 for value in control["finger_flex"].values())


def test_unknown_gesture_keeps_measurements_but_is_not_command_ready():
    control = build_control_representation(
        _payload(
            "unknown",
            0.73,
            0.12,
            {"thumb": 0.03, "index": 0.22, "middle": 0.03, "ring": 0.02, "little": 0.02},
        ),
        _control_cfg(),
    )

    assert control["features_valid"] is True
    assert control["command_ready"] is False
    assert control["valid"] is False
    assert control["preferred_mapping"] is None
    assert control["thumb_index_proximity"] > 0.5
    assert control["effective_pinch_strength"] == 0.0
