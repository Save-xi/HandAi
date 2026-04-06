import logging

import main as app_main
from features.hand_features import extract_hand_features
from gesture.rule_based_gesture import GestureStabilizer, infer_gesture_raw
from output.frame_payload_contract import normalize_frame_payload, validate_frame_payload
from svh.svh_adapter import build_svh_command_preview
from svh.svh_transport_mock import MockSvhTransport
from control.control_representation import build_control_representation


def _baseline_payload(kind, synthetic_hand_pose, gesture_cfg):
    xy, xyz = synthetic_hand_pose(kind)
    payload = extract_hand_features(xy, "Right", 0.98, 123.0, landmarks_xyz=xyz)
    payload["gesture_raw"] = infer_gesture_raw(payload, gesture_cfg)
    stabilizer = GestureStabilizer(confirm_frames=2, unknown_confirm_frames=1)
    payload["gesture_stable"] = stabilizer.update(payload["gesture_raw"])
    payload["gesture_stable"] = stabilizer.update(payload["gesture_raw"])
    return payload


def _run_pipeline(kind, synthetic_hand_pose, gesture_cfg, control_cfg, svh_cfg):
    payload = _baseline_payload(kind, synthetic_hand_pose, gesture_cfg)
    payload["control_representation"] = build_control_representation(payload, control_cfg)
    payload["control_ready"] = bool(payload["control_representation"]["command_ready"])
    payload["svh_preview"] = build_svh_command_preview(payload, svh_cfg)
    payload["frame_index"] = 12
    payload["fps"] = 30.0
    payload["latency_ms"] = 18.5
    payload = normalize_frame_payload(payload, include_deprecated_aliases=False)
    return payload


def test_integration_pipeline_open_pose_reaches_valid_grasp_preview(synthetic_hand_pose, gesture_cfg, control_cfg, svh_cfg):
    payload = _run_pipeline("open", synthetic_hand_pose, gesture_cfg, control_cfg, svh_cfg)
    transport = MockSvhTransport()
    result = transport.send(payload["svh_preview"])

    assert payload["gesture_raw"] == "open"
    assert payload["gesture_stable"] == "open"
    assert payload["control_representation"]["preferred_mapping"] == "grasp"
    assert payload["control_ready"] is True
    assert payload["svh_preview"]["valid"] is True
    assert max(payload["svh_preview"]["target_positions"]) < 0.1
    assert payload["control_representation"]["valid"] == payload["control_representation"]["command_ready"]
    assert result["accepted"] is True
    assert result["recorded_count"] == 1
    assert validate_frame_payload(payload) == []


def test_integration_pipeline_fist_pose_reaches_closed_grasp_preview(synthetic_hand_pose, gesture_cfg, control_cfg, svh_cfg):
    payload = _run_pipeline("fist", synthetic_hand_pose, gesture_cfg, control_cfg, svh_cfg)
    transport = MockSvhTransport()
    transport.send(payload["svh_preview"])

    assert payload["gesture_raw"] == "fist"
    assert payload["gesture_stable"] == "fist"
    assert payload["control_representation"]["preferred_mapping"] == "grasp"
    assert payload["control_representation"]["grasp_close"] > 0.5
    assert payload["svh_preview"]["valid"] is True
    assert min(payload["svh_preview"]["target_positions"][1:]) > 0.5
    assert transport.last_command == payload["svh_preview"]


def test_integration_pipeline_pinch_pose_reaches_valid_pinch_preview(synthetic_hand_pose, gesture_cfg, control_cfg, svh_cfg):
    payload = _run_pipeline("pinch", synthetic_hand_pose, gesture_cfg, control_cfg, svh_cfg)

    assert payload["gesture_raw"] == "pinch"
    assert payload["gesture_stable"] == "pinch"
    assert payload["control_representation"]["preferred_mapping"] == "pinch"
    assert payload["control_representation"]["effective_pinch_strength"] > 0.4
    assert payload["svh_preview"]["valid"] is True
    assert payload["svh_preview"]["target_positions"][0] > payload["svh_preview"]["target_positions"][2]
    assert payload["svh_preview"]["target_positions"][1] > payload["svh_preview"]["target_positions"][2]
    assert all(0.0 <= value <= 1.0 for value in payload["svh_preview"]["target_positions"])


def test_extension_chain_keeps_valid_payload_when_control_extension_fails(
    synthetic_hand_pose,
    gesture_cfg,
    control_cfg,
    caplog,
    monkeypatch,
):
    payload = _baseline_payload("open", synthetic_hand_pose, gesture_cfg)
    runtime = app_main.RuntimeMode(
        gui_enabled=False,
        headless=True,
        input_source_type="webcam",
        input_mirrored=False,
        control_extension_enabled=True,
        svh_preview_enabled=False,
        video_file_path=None,
    )

    def _boom(*_args, **_kwargs):
        raise RuntimeError("control exploded")

    monkeypatch.setattr(app_main, "build_control_representation", _boom)
    logger = logging.getLogger("test_extension_chain_control_failure")

    with caplog.at_level(logging.WARNING):
        diagnostics = app_main._apply_extension_chain(payload, control_cfg, runtime, svh_transport=None, logger=logger)

    payload["frame_index"] = 1
    payload["fps"] = 30.0
    payload["latency_ms"] = 10.0
    payload = normalize_frame_payload(payload, include_deprecated_aliases=False)

    assert diagnostics == [{"extension": "control_representation", "error": "RuntimeError: control exploded"}]
    assert payload["control_representation"]["valid"] is False
    assert payload["control_ready"] is False
    assert validate_frame_payload(payload) == []
    assert "control_representation extension failed" in caplog.text


def test_extension_chain_keeps_valid_payload_when_svh_preview_fails(
    synthetic_hand_pose,
    gesture_cfg,
    control_cfg,
    svh_cfg,
    caplog,
    monkeypatch,
):
    payload = _baseline_payload("pinch", synthetic_hand_pose, gesture_cfg)
    runtime = app_main.RuntimeMode(
        gui_enabled=False,
        headless=True,
        input_source_type="webcam",
        input_mirrored=False,
        control_extension_enabled=True,
        svh_preview_enabled=True,
        video_file_path=None,
    )

    def _boom(*_args, **_kwargs):
        raise ValueError("svh preview exploded")

    monkeypatch.setattr(app_main, "build_svh_command_preview", _boom)
    logger = logging.getLogger("test_extension_chain_svh_failure")

    with caplog.at_level(logging.WARNING):
        diagnostics = app_main._apply_extension_chain(payload, {**control_cfg, **svh_cfg}, runtime, svh_transport=None, logger=logger)

    payload["frame_index"] = 2
    payload["fps"] = 30.0
    payload["latency_ms"] = 11.0
    payload = normalize_frame_payload(payload, include_deprecated_aliases=False)

    assert diagnostics == [{"extension": "svh_preview", "error": "ValueError: svh preview exploded"}]
    assert payload["control_representation"]["command_ready"] is True
    assert payload["control_ready"] is True
    assert payload["svh_preview"]["valid"] is False
    assert validate_frame_payload(payload) == []
    assert "svh_preview extension failed" in caplog.text
