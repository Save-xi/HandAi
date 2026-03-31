from svh.svh_adapter import build_svh_command_preview


def _svh_cfg():
    return {
        "svh_enable_preview": True,
        "svh_enable_gesture_fallback": False,
        "svh_preview_channel_count": 5,
        "svh_preview_mode": "preview",
        "svh_transport": "mock",
        "svh_protocol_sync_bytes": [76, 170],
        "svh_grasp_open_ref": 0.02,
        "svh_grasp_closed_ref": 0.55,
        "svh_pinch_open_ref": 0.45,
        "svh_pinch_closed_ref": 0.08,
        "svh_hand_open_ratio_open_ref": 0.95,
        "svh_hand_open_ratio_closed_ref": 0.25,
        "svh_position_open_value": 0.0,
        "svh_position_closed_value": 1.0,
        "svh_thumb_grasp_scale": 0.85,
        "svh_pinch_support_scale": 0.20,
        "svh_pinch_index_open_ref": 0.05,
        "svh_pinch_index_closed_ref": 0.35,
    }


def _payload(gesture: str, hand_open_ratio, pinch_distance_norm, finger_curl, detected: bool = True):
    return {
        "detected": detected,
        "gesture": gesture,
        "gesture_raw": gesture,
        "hand_open_ratio": hand_open_ratio,
        "pinch_distance_norm": pinch_distance_norm,
        "finger_curl": finger_curl,
    }


def test_svh_invalid_frame_produces_empty_command():
    preview = build_svh_command_preview(
        _payload(
            "unknown",
            None,
            None,
            {"thumb": None, "index": None, "middle": None, "ring": None, "little": None},
            detected=False,
        ),
        _svh_cfg(),
    )

    assert preview["enabled"] is True
    assert preview["valid"] is False
    assert preview["target_channels"] == []
    assert preview["target_positions"] == []


def test_open_and_fist_map_to_different_grasp_preview_ranges():
    cfg = _svh_cfg()
    open_preview = build_svh_command_preview(
        _payload(
            "open",
            0.95,
            0.82,
            {"thumb": 0.03, "index": 0.02, "middle": 0.02, "ring": 0.02, "little": 0.02},
        ),
        cfg,
    )
    fist_preview = build_svh_command_preview(
        _payload(
            "fist",
            0.24,
            0.28,
            {"thumb": 0.14, "index": 0.53, "middle": 0.57, "ring": 0.51, "little": 0.44},
        ),
        cfg,
    )

    assert open_preview["valid"] is True
    assert fist_preview["valid"] is True
    assert max(open_preview["target_positions"]) < min(fist_preview["target_positions"][1:])


def test_pinch_preview_prioritizes_thumb_and_index_channels():
    preview = build_svh_command_preview(
        _payload(
            "pinch",
            0.83,
            0.12,
            {"thumb": 0.04, "index": 0.18, "middle": 0.02, "ring": 0.02, "little": 0.02},
        ),
        _svh_cfg(),
    )

    assert preview["valid"] is True
    assert preview["command_source"] == "control_representation"
    assert preview["target_positions"][0] > preview["target_positions"][2]
    assert preview["target_positions"][1] > preview["target_positions"][2]


def test_known_gesture_does_not_fall_back_when_fallback_is_disabled():
    preview = build_svh_command_preview(
        _payload(
            "open",
            None,
            None,
            {"thumb": None, "index": None, "middle": None, "ring": None, "little": None},
        ),
        _svh_cfg(),
    )

    assert preview["valid"] is False
    assert preview["command_source"] is None
    assert preview["target_channels"] == []
    assert preview["target_positions"] == []


def test_known_gesture_can_fall_back_when_explicitly_enabled():
    cfg = _svh_cfg()
    cfg["svh_enable_gesture_fallback"] = True
    preview = build_svh_command_preview(
        _payload(
            "open",
            None,
            None,
            {"thumb": None, "index": None, "middle": None, "ring": None, "little": None},
        ),
        cfg,
    )

    assert preview["valid"] is True
    assert preview["command_source"] == "gesture_fallback"


def test_unknown_gesture_with_measurements_does_not_emit_command():
    preview = build_svh_command_preview(
        _payload(
            "unknown",
            0.73,
            0.12,
            {"thumb": 0.03, "index": 0.22, "middle": 0.03, "ring": 0.02, "little": 0.02},
        ),
        _svh_cfg(),
    )

    assert preview["valid"] is False
    assert preview["target_channels"] == []
    assert preview["target_positions"] == []
