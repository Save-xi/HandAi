import json
from pathlib import Path

from output.frame_payload_contract import (
    DEPRECATED_ALIASES,
    FRAME_PAYLOAD_REQUIRED_FIELDS,
    normalize_frame_payload,
    validate_frame_payload,
)
from output.json_exporter import JsonExporter

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _landmarks_2d():
    return [[0.1 + index * 0.01, 0.2 + index * 0.01] for index in range(21)]


def _landmarks_3d():
    return [[x, y, -index * 0.001] for index, (x, y) in enumerate(_landmarks_2d())]


def _sample_payload():
    return normalize_frame_payload(
        {
            "timestamp": 1.0,
            "frame_index": 7,
            "detected": True,
            "handedness": "Right",
            "confidence": 0.95,
            "control_ready": True,
            "gesture_raw": "open",
            "gesture_stable": "open",
            "pinch_distance_norm": 0.1,
            "hand_open_ratio": 0.7,
            "finger_curl": {"thumb": 0.1, "index": 0.1, "middle": 0.1, "ring": 0.1, "little": 0.1},
            "landmarks_2d": _landmarks_2d(),
            "landmarks_3d": _landmarks_3d(),
            "control_representation": {
                "valid": True,
                "features_valid": True,
                "command_ready": True,
                "source": "features",
                "gesture_context": "open",
                "preferred_mapping": "grasp",
                "grasp_close": 0.1,
                "thumb_index_proximity": 0.2,
                "effective_pinch_strength": 0.0,
                "pinch_strength": 0.0,
                "support_flex": 0.1,
                "finger_flex": {"thumb": 0.1, "index": 0.1, "middle": 0.1, "ring": 0.1, "little": 0.1},
            },
            "svh_preview": {
                "enabled": True,
                "mode": "preview",
                "valid": True,
                "command_source": "control_representation",
                "target_channels": [0, 1, 2, 3, 4],
                "target_positions": [0.1, 0.2, 0.3, 0.4, 0.5],
                "target_ticks_preview": [],
                "protocol_hint": {
                    "set_control_state_addr": "0x09",
                    "set_all_channels_addr": "0x03",
                    "transport": "mock",
                    "channel_layout": "compact5",
                    "channel_order": "thumb,index,middle,ring,little",
                    "position_units": "normalized_preview",
                    "target_tick_units": "none",
                },
            },
            "fps": 30.0,
            "latency_ms": 10.0,
        },
        include_deprecated_aliases=False,
    )


def test_schema_file_matches_frozen_contract():
    schema = json.loads((PROJECT_ROOT / "schemas" / "frame_payload.schema.json").read_text(encoding="utf-8"))

    assert schema["required"] == list(FRAME_PAYLOAD_REQUIRED_FIELDS)
    assert schema["deprecatedAliases"] == DEPRECATED_ALIASES
    assert schema["properties"]["gesture_stable"]["type"] == "string"
    assert schema["properties"]["svh_preview"]["type"] == "object"
    assert schema["properties"]["landmarks_3d"]["type"] == "array"
    assert schema["allOf"][0]["then"]["properties"]["landmarks_2d"]["minItems"] == 21
    assert schema["allOf"][0]["then"]["properties"]["landmarks_3d"]["maxItems"] == 21


def test_normalizer_accepts_deprecated_aliases_but_does_not_reemit_them():
    payload = normalize_frame_payload(
        {
            "timestamp": 1.0,
            "frame_index": 1,
            "detected": False,
            "handedness": None,
            "confidence": None,
            "gesture_raw": "unknown",
            "gesture": "unknown",
            "pinch_distance_norm": None,
            "hand_open_ratio": None,
            "finger_curl": {"thumb": None, "index": None, "middle": None, "ring": None, "little": None},
            "landmarks_2d": [],
            "control_representation": {
                "valid": False,
                "features_valid": False,
                "command_ready": False,
                "source": None,
                "gesture_context": None,
                "preferred_mapping": None,
                "grasp_close": None,
                "thumb_index_proximity": None,
                "effective_pinch_strength": None,
                "pinch_strength": None,
                "support_flex": None,
                "finger_flex": {"thumb": None, "index": None, "middle": None, "ring": None, "little": None},
            },
            "svh": {
                "enabled": True,
                "mode": "preview",
                "valid": False,
                "command_source": None,
                "target_channels": [],
                "target_positions": [],
                "target_ticks_preview": [],
                "protocol_hint": {
                    "set_control_state_addr": "0x09",
                    "set_all_channels_addr": "0x03",
                    "transport": "mock",
                    "channel_layout": "compact5",
                    "channel_order": "thumb,index,middle,ring,little",
                    "position_units": "normalized_preview",
                    "target_tick_units": "none",
                },
            },
            "control_ready": False,
            "fps": 0.0,
            "latency_ms": 0.0,
        },
        include_deprecated_aliases=False,
    )

    assert payload["gesture_stable"] == "unknown"
    assert "gesture" not in payload
    assert "svh" not in payload
    assert payload["svh_preview"]["valid"] is False
    assert validate_frame_payload(payload) == []


def test_normalizer_repairs_misaligned_landmarks_3d_to_match_landmarks_2d():
    payload = normalize_frame_payload(
        {
            "timestamp": 1.0,
            "frame_index": 2,
            "detected": True,
            "handedness": "Right",
            "confidence": 0.9,
            "control_ready": False,
            "gesture_raw": "unknown",
            "gesture_stable": "unknown",
            "pinch_distance_norm": None,
            "hand_open_ratio": None,
            "finger_curl": {"thumb": None, "index": None, "middle": None, "ring": None, "little": None},
            "landmarks_2d": _landmarks_2d(),
            "landmarks_3d": [[0.1, 0.2, -0.1]],
            "control_representation": {
                "valid": False,
                "features_valid": False,
                "command_ready": False,
                "source": None,
                "gesture_context": None,
                "preferred_mapping": None,
                "grasp_close": None,
                "thumb_index_proximity": None,
                "effective_pinch_strength": None,
                "pinch_strength": None,
                "support_flex": None,
                "finger_flex": {"thumb": None, "index": None, "middle": None, "ring": None, "little": None},
            },
            "svh_preview": {
                "enabled": False,
                "mode": "disabled",
                "valid": False,
                "command_source": None,
                "target_channels": [],
                "target_positions": [],
                "target_ticks_preview": [],
                "protocol_hint": {
                    "set_control_state_addr": "0x09",
                    "set_all_channels_addr": "0x03",
                    "transport": "mock",
                    "channel_layout": "compact5",
                    "channel_order": "thumb,index,middle,ring,little",
                    "position_units": "normalized_preview",
                    "target_tick_units": "none",
                },
            },
            "fps": 0.0,
            "latency_ms": 0.0,
        },
        include_deprecated_aliases=False,
    )

    assert len(payload["landmarks_3d"]) == len(payload["landmarks_2d"]) == 21
    assert validate_frame_payload(payload) == []


def test_detected_frame_rejects_partial_or_extra_landmarks():
    payload = _sample_payload()

    payload["landmarks_2d"] = payload["landmarks_2d"][:20]
    payload["landmarks_3d"] = payload["landmarks_3d"][:20]
    errors = validate_frame_payload(payload)
    assert "detected=true 时 landmarks_2d 必须恰好包含 21 个点" in errors
    assert "detected=true 时 landmarks_3d 必须恰好包含 21 个点" in errors

    payload["landmarks_2d"] = _landmarks_2d() + [[0.5, 0.5]]
    payload["landmarks_3d"] = _landmarks_3d() + [[0.5, 0.5, 0.0]]
    errors = validate_frame_payload(payload)
    assert "detected=true 时 landmarks_2d 必须恰好包含 21 个点" in errors
    assert "detected=true 时 landmarks_3d 必须恰好包含 21 个点" in errors


def test_no_detection_frame_rejects_stale_landmarks_and_ready_state():
    payload = _sample_payload()
    payload["detected"] = False
    payload["handedness"] = None
    payload["confidence"] = None
    payload["control_ready"] = True
    payload["svh_preview"]["valid"] = False
    payload["svh_preview"]["command_source"] = None
    payload["svh_preview"]["target_channels"] = []
    payload["svh_preview"]["target_positions"] = []

    errors = validate_frame_payload(payload)
    assert "detected=false 时 landmarks_2d 必须为空" in errors
    assert "detected=false 时 landmarks_3d 必须为空" in errors
    assert "detected=false 时 control_ready 不能为 true" in errors


def test_optional_timing_v1_accepts_finite_epoch_milliseconds():
    payload = _sample_payload()
    payload["timing"] = {
        "schema_version": 1,
        "clock": "unix_epoch_ms",
        "source_read_start_unix_ms": 1_000_000.0,
        "source_read_end_unix_ms": 1_000_001.0,
        "detection_end_unix_ms": 1_000_005.0,
        "baseline_end_unix_ms": 1_000_006.0,
        "preview_end_unix_ms": 1_000_007.0,
        "payload_ready_unix_ms": 1_000_008.0,
        "udp_send_attempt_unix_ms": None,
    }

    assert validate_frame_payload(payload) == []

    payload["timing"]["udp_send_attempt_unix_ms"] = 1_000_009.0
    assert validate_frame_payload(payload) == []


def test_timing_rejects_non_finite_and_unknown_values():
    payload = _sample_payload()
    payload["timing"] = {
        "schema_version": 1,
        "clock": "unix_epoch_ms",
        "source_read_start_unix_ms": float("nan"),
        "source_read_end_unix_ms": 1_000_001.0,
        "detection_end_unix_ms": 1_000_005.0,
        "baseline_end_unix_ms": 1_000_006.0,
        "preview_end_unix_ms": 1_000_007.0,
        "payload_ready_unix_ms": 1_000_008.0,
        "udp_send_attempt_unix_ms": float("inf"),
        "unexpected": 1.0,
    }

    errors = validate_frame_payload(payload)

    assert "timing 不允许未知字段：unexpected" in errors
    assert "timing.source_read_start_unix_ms 必须是有限数字" in errors
    assert "timing.udp_send_attempt_unix_ms 必须是有限数字" in errors


def test_json_exporter_persists_canonical_payload(tmp_path):
    obj = _sample_payload()
    p = tmp_path / "last.json"
    jsonl_path = tmp_path / "session.jsonl"
    ex = JsonExporter(str(p), save_last_json=True, jsonl_path=str(jsonl_path))
    ex.save_last_frame(obj)
    ex.append_jsonl(obj)
    content = ex.to_json_str(obj)
    console_obj = ex.to_console_obj(obj, landmarks_preview_count=2)
    saved = json.loads(p.read_text(encoding="utf-8"))
    saved_jsonl = json.loads(jsonl_path.read_text(encoding="utf-8").splitlines()[0])

    assert validate_frame_payload(saved) == []
    assert validate_frame_payload(saved_jsonl) == []
    assert "\"gesture_stable\"" in content
    assert "\"svh_preview\"" in content
    assert "\"gesture\"" not in content
    assert "\"svh\"" not in content
    assert p.exists()
    assert jsonl_path.exists()
    assert console_obj["landmarks_count"] == 21
    assert console_obj["landmarks_2d_preview"] == [[0.1, 0.2], [0.11, 0.21]]
    assert console_obj["landmarks_3d_count"] == 21
    assert console_obj["landmarks_3d_preview"] == [[0.1, 0.2, 0.0], [0.11, 0.21, -0.001]]
    assert console_obj["control_representation"]["valid"] is True
    assert console_obj["svh_preview"]["target_positions_count"] == 5
    assert console_obj["svh_preview"]["target_positions_preview"] == [0.1, 0.2]
    assert console_obj["svh_preview"]["target_ticks_count"] == 0
    assert console_obj["svh_preview"]["target_ticks_preview_short"] == []
    assert console_obj["svh_preview"]["protocol_hint"]["channel_layout"] == "compact5"


def test_example_payloads_follow_frozen_contract():
    for name in ["sample_output.json", "sample_output_svh_9ch.json"]:
        payload = json.loads((PROJECT_ROOT / "examples" / name).read_text(encoding="utf-8"))
        assert validate_frame_payload(payload) == []


def test_sample_session_jsonl_lines_follow_frozen_contract():
    lines = (PROJECT_ROOT / "examples" / "sample_session.jsonl").read_text(encoding="utf-8").splitlines()

    assert len(lines) >= 2
    for line in lines:
        payload = json.loads(line)
        assert validate_frame_payload(payload) == []
