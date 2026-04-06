from control.control_representation import build_control_representation, empty_control_representation
from output.frame_payload_contract import FINGER_NAMES, validate_frame_payload
from svh.svh_adapter import build_svh_command_preview, empty_svh_preview


def _empty_finger_map():
    return {name: None for name in FINGER_NAMES}


def _svh_cfg():
    return {
        "svh_enable_preview": True,
        "svh_enable_gesture_fallback": False,
        "svh_preview_layout": "compact5",
        "svh_preview_channel_count": 5,
        "svh_preview_mode": "preview",
        "svh_transport": "mock",
        "svh_grasp_open_ref": 0.02,
        "svh_grasp_closed_ref": 0.55,
        "svh_pinch_open_ref": 0.45,
        "svh_pinch_closed_ref": 0.08,
        "svh_hand_open_ratio_open_ref": 0.95,
        "svh_hand_open_ratio_closed_ref": 0.25,
        "svh_position_open_value": 0.0,
        "svh_position_closed_value": 1.0,
        "svh_thumb_grasp_scale": 0.85,
        "svh_thumb_opposition_scale": 0.75,
        "svh_pinch_support_scale": 0.20,
        "svh_open_spread_scale": 0.25,
        "svh_grasp_spread_scale": 0.05,
        "svh_pinch_spread_scale": 0.10,
        "svh_pinch_index_open_ref": 0.05,
        "svh_pinch_index_closed_ref": 0.35,
    }


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


def test_invalid_control_representation_matches_canonical_empty_builder():
    payload = {
        "detected": False,
        "gesture_raw": "unknown",
        "gesture_stable": "unknown",
        "hand_open_ratio": None,
        "pinch_distance_norm": None,
        "finger_curl": _empty_finger_map(),
    }

    assert build_control_representation(payload, _control_cfg()) == empty_control_representation()


def test_invalid_svh_preview_matches_canonical_empty_builder():
    cfg = _svh_cfg()
    payload = {
        "detected": False,
        "gesture_raw": "unknown",
        "gesture_stable": "unknown",
        "hand_open_ratio": None,
        "pinch_distance_norm": None,
        "finger_curl": _empty_finger_map(),
    }

    assert build_svh_command_preview(payload, cfg) == empty_svh_preview(cfg, enabled=True, mode="preview")


def test_canonical_empty_builders_fit_frozen_contract():
    payload = {
        "timestamp": 1.0,
        "frame_index": 0,
        "detected": False,
        "handedness": None,
        "confidence": None,
        "control_ready": False,
        "gesture_raw": "unknown",
        "gesture_stable": "unknown",
        "pinch_distance_norm": None,
        "hand_open_ratio": None,
        "finger_curl": _empty_finger_map(),
        "landmarks_2d": [],
        "landmarks_3d": [],
        "control_representation": empty_control_representation(),
        "svh_preview": empty_svh_preview(_svh_cfg(), enabled=False, mode="disabled"),
        "fps": 0.0,
        "latency_ms": 0.0,
    }

    assert validate_frame_payload(payload) == []
