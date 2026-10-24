"""映射问题 M1/M2/M7/M6/M8 的受控姿态回归；不是人工精度验收。"""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from perception.base import HandDetection
from output.frame_payload_contract import validate_frame_payload
from control.control_representation import build_control_representation
from control.adaptive_flex import AdaptiveFlexState
from pipeline import HandPipeline
from svh.mapping_contract import assert_mapping_compatible, mapping_contract_payload
from svh.svh_adapter import build_svh_command_preview
from utils.config import load_config, validate_config

POSE = json.loads((Path(__file__).parent / "fixtures/mapping_controlled_pose.json").read_text(encoding="utf-8"))


def controlled_geometry():
    points = np.asarray(POSE["raw_xyz"], dtype=float).copy()
    points[:, 1] *= POSE["height"] / POSE["width"]
    return points


def rotate_chain(points, indices, pivot, axis, angle):
    result = points.copy()
    axis = np.asarray(axis) / np.linalg.norm(axis)
    offset = result[indices] - pivot
    result[indices] = pivot + offset * math.cos(angle) + np.cross(axis, offset) * math.sin(angle) + np.outer(offset @ axis, axis) * (1 - math.cos(angle))
    return result


def run_pose(config, points=None):
    points = controlled_geometry() if points is None else points
    raw = points.copy()
    raw[:, 1] *= POSE["width"] / POSE["height"]
    detection = HandDetection(raw[:, :2].tolist(), raw.tolist(), "Right", 0.98,
                              POSE["width"], POSE["height"], "mediapipe_image_xyz")
    pipe = HandPipeline(load_config(config))
    for i in range(4):
        output = pipe.process_detections([detection], frame_index=i, timestamp=1 + i / 30)
    return output


def single_finger_case(kind):
    points = controlled_geometry()
    normal = np.cross(points[5] - points[0], points[17] - points[0])
    normal /= np.linalg.norm(normal)
    if kind.startswith("index"):
        along = points[6] - points[5]
        along /= np.linalg.norm(along)
        axis = points[17] - points[5]
        axis -= (axis @ along) * along
        test = rotate_chain(points, [8], points[5], axis, math.radians(10))
        sign = 1 if (test[8] - points[5]) @ normal >= (points[8] - points[5]) @ normal else -1
        return rotate_chain(points, [6, 7, 8], points[5], axis, sign * math.radians(int(kind[5:])))
    direction = points[4] - points[1]
    direction /= np.linalg.norm(direction)
    target = points[17] - points[1]
    target = target / np.linalg.norm(target) + 0.6 * normal
    target /= np.linalg.norm(target)
    return rotate_chain(points, [2, 3, 4], points[1], np.cross(direction, target), math.acos(np.clip(direction @ target, -1, 1)))


@pytest.mark.parametrize("kind", ["index45", "index70", "thumb"])
def test_m2_single_finger_motion_does_not_rescale_other_finger_features(kind):
    base = run_pose("configs/ai_mapping_m2.yaml")
    changed = run_pose("configs/ai_mapping_m2.yaml", single_finger_case(kind))
    unchanged = ("middle", "ring", "little") if kind.startswith("index") else ("index", "middle", "ring", "little")
    assert changed["input_diagnostics"]["release_weight"] == 0
    for name in unchanged:
        assert changed["control_representation"]["finger_flex"][name] == pytest.approx(base["control_representation"]["finger_flex"][name], abs=1e-12)


@pytest.mark.parametrize("kind", ["index45", "thumb"])
def test_m2_preview_channels_follow_local_finger_features(kind):
    base = run_pose("configs/ai_mapping_m2.yaml")
    changed = run_pose("configs/ai_mapping_m2.yaml", single_finger_case(kind))
    channels = range(4, 8) if kind.startswith("index") else range(2, 8)
    delta = np.abs(np.subtract(changed["svh_preview"]["target_positions"], base["svh_preview"]["target_positions"]))
    assert max(delta[list(channels)]) < 0.05
    if kind == "thumb":
        assert max(delta[:2]) > 0.05


@pytest.mark.parametrize("kind", ["index45", "index70", "thumb"])
def test_m7_controlled_motion_no_longer_switches_on_projected_pinch(kind):
    base = run_pose("configs/ai_mapping_m7.yaml")
    points = single_finger_case(kind)
    changed = run_pose("configs/ai_mapping_m7.yaml", points)
    assert changed["control_ready"] and changed["svh_preview"]["valid"]
    if kind.startswith("index"):
        assert changed["gesture_stable"] != "pinch"
        assert changed["control_representation"]["preferred_mapping"] == "grasp"
        expected = np.linalg.norm(points[4] - points[8]) / np.linalg.norm(points[0, :2] - points[9, :2])
        assert changed["pinch_distance_norm"] == pytest.approx(expected)
        channels = list(range(4, 8))
    else:
        channels = list(range(2, 8))
    delta = np.abs(np.subtract(changed["svh_preview"]["target_positions"], base["svh_preview"]["target_positions"]))
    assert max(delta[channels]) < 0.05
    if kind == "thumb":
        assert max(delta[:2]) > 0.05


def test_m7_close_thumb_and_index_with_same_depth_still_produce_pinch(synthetic_hand_pose):
    from test_m3a_geometry import encoded
    d = encoded(synthetic_hand_pose("pinch")[1])
    pipe = HandPipeline(load_config("configs/ai_mapping_m7.yaml"))
    for i in range(3):
        out = pipe.process_detections([d], frame_index=i, timestamp=1 + i / 30)
    assert out["gesture_stable"] == "pinch" and out["svh_preview"]["valid"]


def preview_state(strength, grasp, flex, mapping="grasp", gesture="unknown"):
    values = dict.fromkeys(("thumb", "index", "middle", "ring", "little"), flex)
    return {"detected": True, "gesture_stable": gesture, "control_representation": {
        "valid": True, "features_valid": True, "command_ready": True, "source": "features",
        "gesture_context": gesture, "preferred_mapping": mapping, "grasp_close": grasp,
        "thumb_index_proximity": strength, "effective_pinch_strength": strength, "pinch_strength": strength,
        "support_flex": flex, "finger_flex": values}}


@pytest.mark.parametrize("grasp", [0.0, 0.1, 0.25, 0.8, 1.0])
@pytest.mark.parametrize("flex", [0.0, 0.3, 1.0])
def test_m6_strength_scan_and_label_flip_are_continuous(grasp, flex):
    cfg = load_config("configs/ai_mapping_m6.yaml")
    scan = []
    for i in range(25, 46):
        first = build_svh_command_preview(preview_state(i/100, grasp, flex), cfg)["target_positions"]
        flipped = build_svh_command_preview(preview_state(i/100, grasp, flex, "pinch", "pinch"), cfg)["target_positions"]
        assert first == flipped
        assert len(first) == 9 and all(0 <= x <= 1 for x in first)
        scan.append(first)
    assert np.abs(np.diff(scan, axis=0)).max() < 0.05


def test_m6_reaches_pinch_endpoint_in_the_current_call():
    cfg = load_config("configs/ai_mapping_m6.yaml")
    previous_cfg = load_config("configs/ai_mapping_m7.yaml")
    expected = build_svh_command_preview(preview_state(0.8, 0.1, 0.2, "pinch"), previous_cfg)
    current = build_svh_command_preview(preview_state(0.8, 0.1, 0.2, "grasp"), cfg)
    assert current == expected


@pytest.mark.parametrize("kind", ["index45", "index70", "thumb"])
def test_m6_keeps_controlled_other_fingers_stable(kind):
    base = run_pose("configs/ai_mapping_m6.yaml")
    changed = run_pose("configs/ai_mapping_m6.yaml", single_finger_case(kind))
    channels = list(range(4, 8)) if kind.startswith("index") else list(range(2, 8))
    delta = np.abs(np.subtract(changed["svh_preview"]["target_positions"], base["svh_preview"]["target_positions"]))
    assert max(delta[channels]) < 0.05


def test_m8_configured_closed_endpoints_reach_long_finger_range():
    cfg = load_config("configs/ai_mapping_m8.yaml")
    assert (cfg["svh_thumb_grasp_scale"], cfg["svh_thumb_opposition_scale"]) == (0.85, 0.75)
    payload = {"detected": True, "timestamp": 1.0, "gesture_stable": "fist", "hand_open_ratio": 0.4,
               "pinch_distance_norm": 0.7,
               "finger_curl": {"thumb": 0.7, **cfg["control_finger_closed_refs"]}}
    payload["control_representation"] = build_control_representation(payload, cfg, flex_state=AdaptiveFlexState())
    preview = build_svh_command_preview(payload, cfg)
    values = preview["target_positions"]
    assert min(values[2:8]) >= 0.9
    assert values[0] <= 0.85 and values[1] <= 0.75
    assert preview["target_channels"] == list(range(9))


def test_m8_thumb_limits_apply_only_to_grasp_endpoint():
    cfg = load_config("configs/ai_mapping_m8.yaml")
    grasp = build_svh_command_preview(preview_state(0, 1, 1), cfg)["target_positions"]
    pinch = build_svh_command_preview(preview_state(1, 0.1, 1, "pinch"), cfg)["target_positions"]
    assert grasp[:2] == [0.85, 0.75]
    assert pinch[0] > 0.85 and pinch[1] > 0.75


@pytest.mark.parametrize("kind", ["index45", "index70", "thumb"])
def test_final_profile_preserves_single_finger_independence(kind):
    base = run_pose("configs/ai_mapping_m8.yaml")
    changed = run_pose("configs/ai_mapping_m8.yaml", single_finger_case(kind))
    assert changed["gesture_stable"] != "pinch"
    assert changed["control_ready"] and validate_frame_payload(changed) == []
    channels = list(range(4, 8)) if kind.startswith("index") else list(range(2, 8))
    delta = np.abs(np.subtract(changed["svh_preview"]["target_positions"], base["svh_preview"]["target_positions"]))
    assert max(delta[channels]) < 0.05
    if kind == "thumb":
        assert max(delta[:2]) > 0.05


def test_m1_controlled_open_pose_reduces_tip_bias_for_each_finger():
    original = run_pose("configs/ai_m3a.yaml")
    changed = run_pose("configs/ai_mapping_m1.yaml")
    assert max(original["finger_curl"].values()) > 0.3
    assert all(changed["finger_curl"][f] < original["finger_curl"][f] for f in original["finger_curl"])
    assert max(changed["finger_curl"][f] for f in ("index", "middle", "ring", "little")) < 0.1


def test_m1_compact_fist_does_not_lose_long_finger_channels(synthetic_hand_pose):
    from test_m3a_geometry import encoded
    d = encoded(synthetic_hand_pose("fist")[1])
    outputs = []
    for path in ("configs/ai_m3a.yaml", "configs/ai_mapping_m1.yaml"):
        pipe = HandPipeline(load_config(path))
        for i in range(3):
            out = pipe.process_detections([d], frame_index=i, timestamp=1 + i / 30)
        outputs.append(out["svh_preview"]["target_positions"])
    assert all(b >= a for a, b in zip(outputs[0][2:8], outputs[1][2:8]))


def test_m1_new_mapping_rejects_old_model_contract():
    old = mapping_contract_payload(load_config("configs/ai_m3a.yaml"))
    cfg = load_config("configs/ai_mapping_m1.yaml")
    assert mapping_contract_payload(cfg)["version"] == "svh9-camera-v5-per-finger-curl"
    with pytest.raises(ValueError, match="version"):
        assert_mapping_compatible(cfg, old)


def test_m8_filter_changes_are_included_in_model_and_cache_contract():
    cfg = load_config("configs/ai_mapping_m8.yaml")
    expected = mapping_contract_payload(cfg)
    changed = deepcopy(cfg)
    changed["control_flex_filter_beta"] = .2
    with pytest.raises(ValueError, match="control_flex_filter_beta"):
        assert_mapping_compatible(changed, expected)


@pytest.mark.parametrize("value", [0, -1, float("nan"), True, 1.2])
def test_m1_bad_tip_references_rejected(value):
    cfg = load_config("configs/ai_mapping_m1.yaml")
    cfg = deepcopy(cfg)
    cfg["finger_tip_closed_refs"]["thumb"] = value
    with pytest.raises(ValueError, match="逐指参考"):
        validate_config(cfg)
