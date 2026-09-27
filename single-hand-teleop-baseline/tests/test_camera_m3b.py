"""摄像头预测的因果性、共同参考、状态查询与有界调度检查。"""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments/intent_prediction"))
from intent_prediction.camera_replay import common_references, latest_schedule, run_scenario, summarize  # noqa: E402
from intent_prediction.camera_sequence import read_config, validate_sequence  # noqa: E402
from pipeline import HandPipeline  # noqa: E402
from prediction.camera_baselines import future_mapping, geometry_detection, observe_frame, predict_camera, sample_pose  # noqa: E402
from test_m3a_geometry import encoded  # noqa: E402
from utils.config import load_config  # noqa: E402

NOMINAL = {"name": "nominal", "arrival_delay_cycle_ms": [0], "drop_every": 0, "detection_stall_ms": 0, "prediction_stall_ms": 0}
METHODS = [("hold_channels", 2), ("hold_pose", 2), ("linear_pose", 4)]


@pytest.fixture
def frames(synthetic_hand_pose):
    opened, closed = np.array(synthetic_hand_pose("open")[1]), np.array(synthetic_hand_pose("fist")[1])
    result = []
    for i in range(20):
        pose = (1 - i / 25) * opened + i / 25 * closed
        detection = encoded(pose)
        points = np.array(detection.landmarks_xyz)
        points[:, 1] *= detection.image_height / detection.image_width
        result.append({"video_id": "fixture", "person_id": None, "session_id": None, "role": "development_validation",
            "frame_id": i, "source_time_ms": i * 1000 / 30, "timebase": "media_pts_ms", "timestamp_source": "fixture_pts",
            "image_width": 1920, "image_height": 1080, "input_valid": True, "reason": "ok", "segment_id": 0,
            "geometry_landmarks": points.tolist(), "detection_ms": 5.0, "baseline_mapping_ms": 2.0})
    return result


def anchor_for(frames):
    pipeline, previous = HandPipeline(load_config("configs/ai_m3a.yaml")), None
    for frame in frames:
        payload = observe_frame(pipeline, frame, previous)
        previous = frame
    return pipeline, payload


def test_camera_linear_matches_affine_pose_oracle_and_does_not_read_future(frames):
    cfg = load_config("configs/ai_m3a.yaml")
    refs = common_references(frames, cfg, 30)
    prefix = frames[:9]
    pipeline, current = anchor_for(prefix)
    result = predict_camera(prefix, pipeline, current, method="linear_pose", window=4)
    np.testing.assert_allclose(result["positions"], refs[8]["positions"], atol=1e-8)
    frames[9]["geometry_landmarks"] = [[0, 0, 0]] * 21
    again = predict_camera(prefix, pipeline, current, method="linear_pose", window=4)
    np.testing.assert_array_equal(result["positions"], again["positions"])
    changed_refs = common_references(frames, cfg, 30)
    assert changed_refs[8]["positions"] != refs[8]["positions"]


def test_fractional_query_preserves_all_state_and_later_grid_outputs(frames):
    pipeline, _ = anchor_for(frames[:3])
    points = np.asarray(frames[2]["geometry_landmarks"])
    before = (deepcopy(vars(pipeline.stabilizer)), deepcopy(vars(pipeline.release_state)), pipeline._last_timestamp)
    d = geometry_detection(points, 1920, 1080)
    pipeline.query_detections([d], frame_index=2, timestamp=frames[2]["source_time_ms"] / 1000)
    after = (vars(pipeline.stabilizer), vars(pipeline.release_state), pipeline._last_timestamp)
    assert before == after
    full = future_mapping(pipeline, frames[2], [points] * 5, [points] * 3)
    fewer = future_mapping(pipeline, frames[2], [points] * 5, [points] * 2, horizons=(100.0, 150.0))
    assert full["positions"][1:] == fewer["positions"]
    assert before == (vars(pipeline.stabilizer), vars(pipeline.release_state), pipeline._last_timestamp)


def test_future_mapping_does_not_use_grid_after_query_to_reject_earlier_horizon(frames):
    pipeline, _ = anchor_for(frames[:3])
    points = np.array(frames[2]["geometry_landmarks"])
    good = future_mapping(pipeline, frames[2], [points] * 4 + [None], [points] * 3)
    assert all(p is not None for p in good["positions"])
    bad = future_mapping(pipeline, frames[2], [points, None] + [points] * 3, [points] * 3)
    assert bad["positions"][0] is not None and bad["positions"][1:] == [None, None]


@pytest.mark.parametrize("failure", ["invalid", "segment", "gap", "frame_id"])
def test_sampling_does_not_bridge_missing_or_changed_segments(frames, failure):
    if failure == "invalid":
        frames[1]["input_valid"] = False
    elif failure == "segment":
        frames[1]["segment_id"] = 1
    elif failure == "gap":
        frames[1]["source_time_ms"] = 200
        frames = frames[:2]
    else:
        frames[1]["frame_id"] = 2
    assert sample_pose(frames, 20, segment_id=0) is None


def test_latest_queue_replaces_waiting_work_and_counts_its_real_wait():
    schedule, reasons = latest_schedule([0, 10, 20, 30, 40], [25] * 5)
    assert [i for i, value in enumerate(schedule) if value] == [0, 2, 4]
    assert reasons[1] == reasons[3] == "queue_replaced"
    assert schedule[2] == {"start_ms": 25.0, "end_ms": 50.0, "queue_ms": 5.0}
    assert schedule[4]["end_ms"] == 75


def test_overdue_baseline_and_dropped_work_stay_in_error_denominator(frames):
    cfg = load_config("configs/ai_m3a.yaml")
    refs = common_references(frames, cfg, 30)
    for frame in frames:
        frame["detection_ms"] = 120
    rows = run_scenario(frames, cfg, methods=[("hold_channels", 2)], fps=30, scenario=NOMINAL)["hold_channels"]
    metrics = summarize(rows, refs)["50"]
    assert metrics["source_count"] == len(frames)
    assert metrics["output_count"] == 0 and metrics["no_output_count"] == len(frames)
    assert metrics["reference_count"] > 0 and metrics["reference_without_output"] == metrics["reference_count"]
    assert metrics["penalized_rmse"] == 1 and metrics["conditional_output_rmse"] is None
    assert any(row["fallback_reason"] == "baseline_queue_replaced" for row in rows)


def test_prediction_failure_uses_current_without_losing_reference_denominator(frames, monkeypatch):
    from intent_prediction import camera_replay
    cfg = load_config("configs/ai_m3a.yaml")
    refs = common_references(frames, cfg, 30)
    def failure(*_args, **_kwargs):
        return {"positions": [None] * 3, "gestures": [None] * 3, "reasons": ["invalid_prediction"] * 3,
                "model_ms": 1.0, "postprocess_ms": 2.0, "total_ms": 3.0}
    monkeypatch.setattr(camera_replay, "predict_camera", failure)
    rows = run_scenario(frames, cfg, methods=[("hold_pose", 2)], fps=30, scenario=NOMINAL)["hold_pose"]
    metrics = summarize(rows, refs)["50"]
    assert metrics["fallback_count"] == len(frames) and metrics["prediction_count"] == 0
    assert metrics["reference_without_output"] == 0 and metrics["reference_count"] > 0
    assert all(row["ready_time_ms"] == pytest.approx(row["source_time_ms"] + 10) for row in rows)


def test_all_methods_share_sources_targets_and_provenance(frames):
    cfg = load_config("configs/ai_m3a.yaml")
    rows = run_scenario(frames, cfg, methods=METHODS, fps=30, scenario=NOMINAL, algorithm_only=True)
    signatures = [[(r["frame_id"], r["source_time_ms"], r["target_time_ms"], r["mapping_version"]) for r in values] for values in rows.values()]
    assert signatures[0] == signatures[1] == signatures[2]
    assert any(r["fallback_reason"] == "history_warmup" and r["used_fallback"] for r in rows["linear_pose_w4"])
    assert all(r["ready_time_ms"] == r["source_time_ms"] for values in rows.values() for r in values)
    for values in rows.values():
        json.dumps(values, allow_nan=False)
        assert all(r["target_time_ms"] == r["source_time_ms"] + r["horizon_ms"] for r in values)


def test_input_loss_cancels_inflight_prediction_from_old_epoch(frames, monkeypatch):
    from intent_prediction import camera_replay
    original = camera_replay.predict_camera
    def slow(*args, **kwargs):
        result = original(*args, **kwargs)
        return {**result, "model_ms": 0.0, "postprocess_ms": 100.0, "total_ms": 100.0}
    monkeypatch.setattr(camera_replay, "predict_camera", slow)
    frames[4].update(input_valid=False, reason="no_right_hand", segment_id=-1)
    for frame in frames[5:]:
        frame["segment_id"] = 1
    rows = run_scenario(frames, load_config("configs/ai_m3a.yaml"), methods=[("hold_pose", 2)], fps=30, scenario=NOMINAL)["hold_pose"]
    assert any(row["fallback_reason"] == "history_reset_before_ready" for row in rows)
    assert all(not row["valid"] for row in rows if row["frame_id"] == 4)
    discarded = [row for row in rows if row["fallback_reason"] == "prediction_queue_replaced"]
    assert discarded and all(row["candidate_positions"] is None and not row["prediction_valid"] for row in discarded)


def test_provenance_split_is_checked_before_opening_video(tmp_path):
    config = json.loads((ROOT / "experiments/intent_prediction/configs/camera_m3b.json").read_text(encoding="utf-8"))
    config["runtime_config"] = str(ROOT / "configs/ai_m3a.yaml")
    for video in config["videos"]:
        video.update(person_id="same_person", session_id="same_session")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="同一已知会话"):
        read_config(path)


@pytest.mark.parametrize("damage", ["geometry", "time", "segment", "nonfinite"])
def test_cached_observations_keep_coordinate_time_and_segment_checks(frames, damage):
    cfg = load_config("configs/ai_m3a.yaml")
    for row in frames:
        raw = np.array(row["geometry_landmarks"])
        raw[:, 1] *= row["image_width"] / row["image_height"]
        row.update(raw_landmarks_xyz=raw.tolist(), read_ms=0.0, task_id="camera_mediapipe_geometry_xyz_v1")
    validate_sequence(frames, cfg)
    if damage == "geometry":
        frames[1]["geometry_landmarks"][3][1] += 0.1
    elif damage == "time":
        frames[1]["source_time_ms"] = 0
    elif damage == "segment":
        frames[1].update(input_valid=False, segment_id=-1)
    else:
        frames[1]["detection_ms"] = float("nan")
    with pytest.raises(ValueError):
        validate_sequence(frames, cfg)
