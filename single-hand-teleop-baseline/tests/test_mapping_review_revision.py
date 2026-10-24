"""R1/R2：不靠慢滤波遮住输入噪声，保留明确的拇指端点和单指响应。"""
from copy import deepcopy

import numpy as np
import pytest

from control.control_representation import build_control_representation
from output.frame_payload_contract import validate_frame_payload
from pipeline import HandPipeline
from svh.mapping_contract import mapping_contract_payload, assert_mapping_compatible
from svh.svh_adapter import build_svh_command_preview
from test_adaptive_flex import detection
from test_mapping_repairs import controlled_geometry, single_finger_case, preview_state
from utils.config import load_config, validate_config

CURRENT = "configs/ai_camera_v10.yaml"


@pytest.mark.parametrize("fps", [30, 13])
def test_r1_control_flex_step_reaches_target_in_first_changed_frame(fps):
    cfg = load_config(CURRENT)
    assert cfg["control_flex_filter_mode"] == "none"
    payload = {"detected": True, "timestamp": 0., "gesture_stable": "unknown",
               "hand_open_ratio": 1., "pinch_distance_norm": 1.,
               "finger_curl": dict.fromkeys(("thumb", "index", "middle", "ring", "little"), 0.)}
    first = build_control_representation(payload, cfg)
    payload["timestamp"] = 1/fps
    payload["finger_curl"] = dict.fromkeys(payload["finger_curl"], 1.)
    changed = build_control_representation(payload, cfg)
    assert all(v == 0 for v in first["finger_flex"].values())
    assert all(v >= .8 for v in changed["finger_flex"].values())
    assert changed["finger_flex"] == dict.fromkeys(payload["finger_curl"], 1.)


@pytest.mark.parametrize("fps", [30, 13])
@pytest.mark.parametrize("kind", ["index45", "index70", "thumb"])
def test_r1_full_pipeline_has_no_added_filter_lag_or_cross_finger_release(fps, kind):
    cfg = load_config(CURRENT)
    pipe = HandPipeline(cfg)
    for i in range(5):
        base = pipe.process_detections([detection(controlled_geometry())], frame_index=i, timestamp=1+i/fps)
    changed = detection(single_finger_case(kind))
    actual = pipe.process_detections([changed], frame_index=5, timestamp=1+5/fps)
    fresh = HandPipeline(cfg).process_detections([changed], frame_index=5, timestamp=1+5/fps)
    assert actual["svh_preview"] == fresh["svh_preview"]
    assert pipe.flex_state.timestamp is None
    assert actual["input_diagnostics"]["release_weight"] == 0
    assert validate_frame_payload(actual) == []
    delta = np.abs(np.subtract(actual["svh_preview"]["target_positions"], base["svh_preview"]["target_positions"]))
    other = list(range(4,8)) if kind.startswith("index") else list(range(2,8))
    assert max(delta[other]) < .05
    if kind == "thumb":
        assert max(delta[:2]) > .05
    else:
        assert actual["gesture_stable"] != "pinch"


def test_r2_open_reference_is_broad_and_b_base_thumb_is_below_point_one():
    cfg = load_config(CURRENT)
    assert cfg["finger_tip_open_refs"]["thumb"]-cfg["finger_tip_closed_refs"]["thumb"] >= .3
    assert cfg["thumb_tip_proximity_weight"] < .70
    out = HandPipeline(cfg).process_detections([detection(controlled_geometry())], frame_index=0, timestamp=1)
    assert out["svh_preview"]["target_positions"][1] < .1


def test_r2_grasp_calibration_reaches_limits_without_limiting_pinch():
    cfg = load_config(CURRENT)
    grasp = build_svh_command_preview(preview_state(0, 1, 1), cfg)["target_positions"]
    pinch = build_svh_command_preview(preview_state(1, .1, 1, "pinch"), cfg)["target_positions"]
    assert grasp[:2] == [.85, .75]
    assert pinch[0] > .85 and pinch[1] > .75
    for grasp_close in (.65, .8, 1.):
        output = build_svh_command_preview(preview_state(0, grasp_close, .2), cfg)["target_positions"]
        assert output[0] >= .9*.85 and output[1] >= .9*.75


def test_r2_keeps_m6_same_frame_blend_continuous_and_label_independent():
    cfg = load_config(CURRENT)
    for grasp in (0., .1, .25, .8, 1.):
        scan = []
        for i in range(25, 46):
            a = build_svh_command_preview(preview_state(i/100, grasp, .3), cfg)["target_positions"]
            b = build_svh_command_preview(preview_state(i/100, grasp, .3, "pinch", "pinch"), cfg)["target_positions"]
            assert a == b
            scan.append(a)
        assert np.abs(np.diff(scan, axis=0)).max() < .05


def test_v10_does_not_reuse_v9_model_or_cache_contract():
    cfg = load_config(CURRENT)
    old = mapping_contract_payload(load_config("configs/ai_mapping_m8.yaml"))
    assert mapping_contract_payload(cfg)["version"] == "svh9-camera-v10-low-lag-thumb"
    with pytest.raises(ValueError, match="version"):
        assert_mapping_compatible(cfg, old)
    for key in ("thumb_tip_proximity_weight", "svh_thumb_grasp_guide_weight"):
        changed = deepcopy(cfg)
        changed[key] = .1
        with pytest.raises(ValueError, match=key):
            assert_mapping_compatible(changed, mapping_contract_payload(cfg))


@pytest.mark.parametrize("key, value", [
    ("thumb_tip_proximity_weight", float("nan")), ("thumb_tip_proximity_weight", True),
    ("svh_thumb_grasp_open_ref", .9), ("svh_thumb_grasp_guide_weight", 1.1),
    ("svh_thumb_control_mode", "direct_opposition"),
])
def test_v10_bad_thumb_configuration_is_rejected(key, value):
    cfg = load_config(CURRENT)
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_old_versions_cannot_silently_accept_new_thumb_weight():
    cfg = load_config("configs/ai_mapping_m8.yaml")
    cfg["thumb_tip_proximity_weight"] = .25
    with pytest.raises(ValueError, match="camera_mapping_stage"):
        validate_config(cfg)
