"""M3-A 坐标变换、不完整输入和跨坏帧状态的回归测试。"""
from copy import deepcopy
import json

import numpy as np
import pytest

from output.frame_payload_contract import validate_frame_payload
from perception.base import HandDetection
from perception.geometry import prepare_geometry
from perception.landmark_quality import assess_control_readiness
from pipeline import HandPipeline
from prediction.shadow_predictor import PredictionShadow
from svh.mapping_contract import assert_mapping_compatible, legacy_v2_mapping_contract, mapping_contract_payload
from utils.config import load_config


def encoded(points, width=1920, height=1080, rotation=0, scale=0.04):
    xyz = np.array(points, dtype=float) * scale
    angle = np.deg2rad(rotation)
    xyz[:, :2] = xyz[:, :2] @ np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    xyz[:, 0] += 0.5
    xyz[:, 1] += 0.5 * height / width
    xyz[:, 1] *= width / height
    return HandDetection(xyz[:, :2].tolist(), xyz.tolist(), "Right", 0.99, width, height, "mediapipe_image_xyz")


def process(detection, cfg=None):
    pipeline = HandPipeline(cfg or load_config("configs/ai_m3a.yaml"))
    for i in range(2):
        output = pipeline.process_detections([detection], frame_index=i, timestamp=1 + i / 30)
    return output


@pytest.mark.parametrize("size", [(1000, 1000), (1920, 1080), (1080, 1920)])
@pytest.mark.parametrize("rotation", [0, 37, 90, 180])
@pytest.mark.parametrize("scale", [0.02, 0.04])
def test_same_geometry_across_aspect_rotation_and_scale(synthetic_hand_pose, size, rotation, scale):
    points = 0.28 * np.array(synthetic_hand_pose("open")[1]) + 0.72 * np.array(synthetic_hand_pose("fist")[1])
    reference = process(encoded(points, 1000, 1000))
    detection = encoded(points, *size, rotation, scale)
    actual = process(detection)
    assert actual["control_ready"] and actual["gesture_raw"] == reference["gesture_raw"]
    for key in ("hand_open_ratio", "pinch_distance_norm"):
        assert actual[key] == pytest.approx(reference[key], abs=1e-8)
    np.testing.assert_allclose(list(actual["finger_curl"].values()), list(reference["finger_curl"].values()), atol=1e-8)
    np.testing.assert_allclose(actual["svh_preview"]["target_positions"], reference["svh_preview"]["target_positions"], atol=1e-8)
    # 绘图/序列化原始点保持原值；几何点的 Y 才乘 H/W。
    np.testing.assert_array_equal(actual["landmarks_2d"], detection.landmarks_2d)
    np.testing.assert_array_equal(actual["landmarks_3d"], detection.landmarks_xyz)
    np.testing.assert_allclose(np.array(actual["input_diagnostics"]["geometry_landmarks"])[:, 1],
                               np.array(detection.landmarks_xyz)[:, 1] * size[1] / size[0])


@pytest.mark.parametrize("damage,reason", [
    ("0", "invalid_xy_shape"), ("5", "invalid_xy_shape"), ("20", "invalid_xy_shape"),
    ("missing_xyz", "invalid_xyz_shape"), ("short_xyz", "invalid_xyz_shape"),
    ("nan", "nonfinite_landmarks"), ("inf", "nonfinite_landmarks"),
    ("coincident", "degenerate_palm"), ("bone", "degenerate_bone"),
    ("size", "missing_or_invalid_image_size"), ("domain", "unsupported_coordinate_space"),
    ("missing_domain", "missing_coordinate_space"),
    ("strings", "nonnumeric_landmarks"),
    ("xy", "inconsistent_xy_xyz"), ("confidence", "invalid_confidence"), ("edge", "out_of_bounds"),
])
def test_invalid_input_immediately_clears_states_and_recovers(synthetic_hand_pose, damage, reason):
    cfg = load_config("configs/ai_m3a.yaml")
    cfg["stable_unknown_consecutive"] = 10  # 丢手不得等待 unknown 确认。
    good = encoded(synthetic_hand_pose("open")[1])
    bad = deepcopy(good)
    if damage.isdigit():
        bad.landmarks_2d = bad.landmarks_2d[:int(damage)]
        bad.landmarks_xyz = bad.landmarks_xyz[:int(damage)]
    elif damage == "missing_xyz":
        bad.landmarks_xyz = None
    elif damage == "short_xyz":
        bad.landmarks_xyz[0] = [0.5, 0.5]
    elif damage in ("nan", "inf"):
        bad.landmarks_xyz[3][2] = float(damage)
    elif damage == "coincident":
        bad.landmarks_xyz = [[0.5, 0.5, 0]] * 21
        bad.landmarks_2d = [[0.5, 0.5]] * 21
    elif damage == "bone":
        bad.landmarks_xyz[7] = bad.landmarks_xyz[6][:]
        bad.landmarks_2d[7] = bad.landmarks_2d[6][:]
    elif damage == "size":
        bad.image_width = None
    elif damage == "domain":
        bad.coordinate_space = "h2o_camera_meters"
    elif damage == "missing_domain":
        bad.coordinate_space = None
    elif damage == "strings":
        bad.landmarks_xyz = [[str(v) for v in p] for p in bad.landmarks_xyz]
        bad.landmarks_2d = [[str(v) for v in p] for p in bad.landmarks_2d]
    elif damage == "xy":
        bad.landmarks_2d[3][0] += 0.01
    elif damage == "confidence":
        bad.confidence = None
    elif damage == "edge":
        for xy, xyz in zip(bad.landmarks_2d, bad.landmarks_xyz):
            xy[0] += 1
            xyz[0] += 1
    pipeline = HandPipeline(cfg)
    shadow = PredictionShadow(predict_fn=lambda h: np.repeat(h[-1:], 3, axis=0), history_frames=3,
        horizon_ms=[50, 100, 150], gate_recent_frames=3, gate_threshold=0.001, gate_temperature=0.01,
        gate_alpha_by_horizon=[0.75] * 3, max_frame_gap_ms=100, device="cpu", model_label="fixture")
    for i, d in enumerate((good, good, bad, good, good)):
        result = pipeline.process_detections([d], frame_index=i, timestamp=1 + i / 30)
        status = shadow.observe(result)
        assert validate_frame_payload(result) == []
        json.dumps(result, allow_nan=False)
        if i == 1:
            assert result["gesture_stable"] == "open" and pipeline.release_state.weight == 1
        elif i == 2:
            assert result["input_diagnostics"]["reason"] == reason
            assert not result["control_ready"] and not result["svh_preview"]["valid"]
            assert result["gesture_stable"] == "unknown"
            assert pipeline.stabilizer.candidate_count == 0 and pipeline.release_state.weight == 0
            assert status["history_frames_available"] == 0
        elif i == 3:
            assert result["control_ready"] and result["gesture_stable"] == "unknown"
            assert status["history_frames_available"] == 1
        elif i == 4:
            assert result["gesture_stable"] == "open"


@pytest.mark.parametrize("count", [0, 5, 20])
def test_quality_entry_does_not_index_short_arrays(count):
    assert not assess_control_readiness([(0.5, 0.5)] * count, {})["control_ready"]


def test_palm_fallback_and_depth_bone_are_not_false_degeneracies(synthetic_hand_pose):
    detection = encoded(synthetic_hand_pose("open")[1])
    detection.landmarks_xyz[9] = detection.landmarks_xyz[0][:]
    detection.landmarks_2d[9] = detection.landmarks_2d[0][:]
    assert process(detection)["control_ready"]
    detection.landmarks_2d[7] = detection.landmarks_2d[6][:]
    detection.landmarks_xyz[7][:2] = detection.landmarks_xyz[6][:2]
    detection.landmarks_xyz[7][2] += 0.03
    assert process(detection)["control_ready"]


def test_image_metadata_mismatch_and_injected_frame_size(synthetic_hand_pose):
    cfg = load_config("configs/ai_m3a.yaml")
    d = encoded(synthetic_hand_pose("open")[1])
    assert prepare_geometry(d, cfg, frame=np.zeros((500, 500, 3)))["reason"] == "image_size_mismatch"
    d.image_height = d.image_width = None
    assert prepare_geometry(d, cfg, frame=np.zeros((1080, 1920, 3)))["input_valid"]
    assert not prepare_geometry(d, cfg)["input_valid"]


@pytest.mark.parametrize("discontinuity", ["gap", "reverse", "resize"])
def test_valid_discontinuity_restarts_gesture_confirmation(synthetic_hand_pose, discontinuity):
    pipeline = HandPipeline(load_config("configs/ai_m3a.yaml"))
    d = encoded(synthetic_hand_pose("open")[1])
    for i in range(2):
        pipeline.process_detections([d], frame_index=i, timestamp=1 + i / 30)
    ts = 1 + 2 / 30
    if discontinuity == "gap":
        ts = 2
    elif discontinuity == "reverse":
        ts = 1
    else:
        d.image_width *= 2
        d.image_height *= 2
    output = pipeline.process_detections([d], frame_index=2, timestamp=ts)
    assert output["input_diagnostics"]["state_reset"] and output["gesture_stable"] == "unknown"
    assert output["control_ready"]  # 当前观测仍有效，不能错误当成丢手。


def test_mapping_version_prevents_old_checkpoint_consuming_new_semantics():
    assert_mapping_compatible(load_config("configs/ai.yaml"), legacy_v2_mapping_contract())
    cfg = load_config("configs/ai_m3a.yaml")
    with pytest.raises(ValueError, match="version"):
        assert_mapping_compatible(cfg, legacy_v2_mapping_contract())
    snapshot = mapping_contract_payload(cfg)
    cfg["control_release_hysteresis"] = 0.02
    with pytest.raises(ValueError, match="control_release_hysteresis"):
        assert_mapping_compatible(cfg, snapshot)


def test_numpy_input_and_input_diagnostics_contract(synthetic_hand_pose):
    d = encoded(synthetic_hand_pose("open")[1])
    d.landmarks_xyz = np.asarray(d.landmarks_xyz)
    d.landmarks_2d = np.asarray(d.landmarks_2d)
    output = process(d)
    assert output["control_ready"] and validate_frame_payload(output) == []
    output["input_diagnostics"]["input_valid"] = False
    assert any("失效输入" in message for message in validate_frame_payload(output))
    output["input_diagnostics"]["geometry_landmarks"][0][2] = float("nan")
    assert any("geometry_landmarks" in message for message in validate_frame_payload(output))
