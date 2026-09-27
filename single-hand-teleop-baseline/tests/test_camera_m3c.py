"""M3-C 人工真值的数值分母、时间和切分边界。"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "experiments/intent_prediction")]
from intent_prediction.camera_m3c import (  # noqa: E402
    aggregate_group, fully_labeled, import_clip, load_timeline, match_event_windows,
    match_events, predicted_events, read_json, save_annotations, score_forecast,
    score_gesture, score_pose, valid_raw_xy, validate_registry,
)
from intent_prediction import camera_m3c  # noqa: E402
from perception.geometry import GEOMETRY_TASK  # noqa: E402
from test_m3a_geometry import encoded  # noqa: E402
from utils.config import load_config  # noqa: E402


def timeline(n=3):
    return [{"frame_id": i, "pts_ms": i * 50.0, "timestamp_source": "container_pts_ms",
             "width": 100, "height": 80} for i in range(n)]


def points():
    return [{"x": 50.0, "y": 40.0, "visible": True} for _ in range(21)]


def test_full_gt_pck_counts_missed_detection_and_uses_original_pixels():
    annotation = {"keypoint_frames": [{"frame_id": 0, "presence": "right", "sampling_purpose": "uniform", "points": points()},
                                      {"frame_id": 1, "presence": "right", "sampling_purpose": "uniform", "points": points()}]}
    rows = [{"raw_landmarks_2d": [[0.5, 0.5]] * 21, "input_valid": False, "reason": "quality_gate"},
            {"raw_landmarks_2d": [], "input_valid": False, "reason": "no_right_hand"}]
    result = score_pose(annotation, rows, timeline(2), 20)
    assert result["visible_gt_points"] == 42
    assert result["full_gt_pck"] == 0.5
    assert result["conditional_mean_pixel_error"] == 0
    assert result["visible_point_coverage"] == 0.5
    assert result["detected_right_frames"] == 1  # 原始检测与控制门控分开
    assert result["by_sampling_purpose"]["uniform"]["full_gt_pck"] == 0.5


def test_unknown_no_output_is_false_negative_and_absent_separate():
    intervals = [{"start_ms": 0, "end_ms": 50, "label": "unknown"},
                 {"start_ms": 50, "end_ms": 100, "label": "absent"}]
    rows = [{"input_valid": False, "gesture_stable": "unknown", "reason": "no_right_hand"},
            {"input_valid": False, "gesture_stable": "unknown", "reason": "no_right_hand"}]
    result = score_gesture(intervals, rows, timeline(2))
    assert result["confusion"]["unknown"]["__no_output__"] == 1
    assert result["per_class"]["unknown"]["recall"] == 0
    assert result["per_class"]["unknown"]["f1"] == 0
    assert result["gt_absent_frames"] == 1
    assert result["false_activation_on_absent_frames"] == 0


def test_event_matching_is_maximum_one_to_one_and_gaps_break_events():
    gt = [(100, "open", "fist"), (200, "open", "fist")]
    predicted = [(160, "open", "fist"), (270, "open", "fist")]
    matched = match_events(gt, predicted, tolerance_ms=80)
    assert matched["matched"] == 2
    assert matched["missed"] == matched["false_trigger"] == 0
    intervals = [{"start_ms": 0, "end_ms": 45, "label": "open"},
                 {"start_ms": 55, "end_ms": 105, "label": "fist"}]
    rows = [{"frame_id": 0, "target_time_ms": 40, "gesture": "open"},
            {"frame_id": 1, "target_time_ms": 60, "gesture": "fist"}]
    assert not fully_labeled(intervals, 40, 60)
    assert predicted_events(rows, intervals) == []


def test_forecast_uses_target_pts_and_separates_base_prediction_lateness():
    intervals = [{"start_ms": 0, "end_ms": 100, "label": "open"},
                 {"start_ms": 100, "end_ms": 250, "label": "fist"}]
    rows = []
    for frame_id, source in enumerate((0.0, 50.0)):
        for horizon in (50, 100, 150):
            target = source + horizon
            rows.append({"frame_id": frame_id, "source_time_ms": source, "target_time_ms": target,
                         "horizon_ms": horizon, "gesture": "fist" if target >= 100 else "open",
                         "valid": True, "prediction_on_time": True, "used_fallback": False,
                         "timing": {"baseline_ready_ms": target + 1 if frame_id == 0 else target - 10},
                         "ready_time_ms": target + 1 if frame_id == 1 else target - 10,
                         "fallback_reason": None})
    result = score_forecast(rows, intervals, 2)
    assert result["50"]["full_gt_gesture_accuracy"] == 1
    assert result["50"]["base_late_count"] == 1
    assert result["50"]["prediction_late_count"] == 1
    assert result["100"]["confusion"]["fist"]["fist"] == 2


def test_annotation_roundtrip_validation_and_session_leakage(tmp_path):
    clip = tmp_path / "clip"
    clip.mkdir()
    (clip / "timeline.jsonl").write_text("".join(json.dumps(row) + "\n" for row in timeline(2)), encoding="utf-8")
    (clip / "manifest.json").write_text(json.dumps({"clip_id": "clip"}), encoding="utf-8")
    annotation = {"schema_version": "camera-m3c-annotation-v1", "clip_id": "clip",
                  "keypoint_frames": [{"frame_id": 0, "presence": "right", "sampling_purpose": "uniform",
                                       "points": points()}],
                  "gesture_intervals": [{"start_ms": 0, "end_ms": 50, "label": "unknown"}],
                  "transition_events": []}
    save_annotations(clip, annotation)
    assert read_json(clip / "annotations.json") == annotation
    broken = json.loads(json.dumps(annotation))
    broken["keypoint_frames"][0]["points"][0]["visible"] = None
    with pytest.raises(ValueError, match="未完成"):
        save_annotations(clip, broken)
    assert read_json(clip / "annotations.json") == annotation
    registry = {"schema_version": "camera-m3c-dataset-v1", "split_policy": "independent_person",
                "clips": [{"person_id": "p", "session_id": "s", "clip_id": "one", "scenario": "x",
                           "role": "development", "evidence_type": "real", "video_path": "a"},
                          {"person_id": "p", "session_id": "s", "clip_id": "two", "scenario": "x",
                           "role": "heldout", "evidence_type": "real", "video_path": "b"}]}
    with pytest.raises(ValueError, match="会话"):
        validate_registry(registry)
    registry["clips"][1]["person_id"] = "q"
    registry["clips"][1]["clip_id"] = "../escape"
    with pytest.raises(ValueError, match="clip_id"):
        validate_registry(registry)


def test_real_video_import_keeps_pts_dimensions_and_synthetic_mark(tmp_path):
    source = tmp_path / "numbered.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 20, (100, 80))
    assert writer.isOpened()
    for i in range(3):
        writer.write(np.full((80, 100, 3), i * 70, dtype=np.uint8))
    writer.release()
    location = import_clip(tmp_path / "dataset", video=source, person_id="synthetic-person",
                           session_id="synthetic-session", clip_id="s1", scenario="synthetic",
                           role="development", split_policy="independent_person", synthetic=True)
    imported = load_timeline(location)
    assert [row["frame_id"] for row in imported] == [0, 1, 2]
    assert all((row["width"], row["height"]) == (100, 80) for row in imported)
    assert imported[1]["pts_ms"] > imported[0]["pts_ms"]
    assert read_json(location / "manifest.json")["evidence_type"] == "synthetic"


def test_raw21_requires_finite_xy_and_sampling_groups_stay_separate():
    assert not valid_raw_xy([[None, None]] * 21)
    assert not valid_raw_xy([[float("nan"), 0.0]] * 21)
    annotation = {"keypoint_frames": [
        {"frame_id": 0, "presence": "right", "sampling_purpose": "uniform", "points": points()},
        {"frame_id": 1, "presence": "right", "sampling_purpose": "failure_targeted", "points": points()},
        {"frame_id": 2, "presence": "absent", "sampling_purpose": "dense_transition", "points": []}]}
    observations = [
        {"raw_landmarks_2d": [[0.5, 0.5]] * 21, "input_valid": True},
        {"raw_landmarks_2d": [[None, None]] * 21, "input_valid": False},
        {"raw_landmarks_2d": [[0.5, 0.5]] * 21, "input_valid": True},
    ]
    pose = score_pose(annotation, observations, timeline(3), 10)
    assert pose["raw21_complete_frames"] == 1
    assert pose["gt_right_frames"] == 2
    assert pose["false_right_frames"] == 1
    assert pose["by_sampling_purpose"]["uniform"]["full_gt_pck"] == 1
    assert pose["by_sampling_purpose"]["failure_targeted"]["full_gt_pck"] == 0


def test_manual_transition_window_lag_and_target_range():
    gt = [{"start_ms": 90, "end_ms": 110, "from": "open", "to": "fist"}]
    assert match_event_windows(gt, [(100, "open", "fist")])["lag_ms"] == [0]
    assert match_event_windows(gt, [(70, "open", "fist")])["lag_ms"] == [-20]
    assert match_event_windows(gt, [(130, "open", "fist")])["lag_ms"] == [20]
    intervals = [{"start_ms": 0, "end_ms": 100, "label": "open"},
                 {"start_ms": 100, "end_ms": 300, "label": "fist"}]
    rows = [{"frame_id": i, "source_time_ms": t-150, "target_time_ms": t, "horizon_ms": 150,
             "gesture": "fist", "valid": True, "prediction_on_time": True, "used_fallback": False,
             "timing": {"baseline_ready_ms": t-10}, "ready_time_ms": t-5, "fallback_reason": None}
            for i, t in enumerate((150, 200))]
    result = score_forecast(rows, intervals, 2, gt)["150"]
    assert result["transition_events_outside_target_range"] == 1
    assert result["transitions"]["missed"] == 0


def test_event_match_cannot_cross_unlabeled_hole():
    intervals = [{"start_ms": 0, "end_ms": 100, "label": "open"},
                 {"start_ms": 100, "end_ms": 180, "label": "fist"},
                 {"start_ms": 200, "end_ms": 400, "label": "unknown"}]
    gt = [{"start_ms": 100, "end_ms": 100, "from": "open", "to": "fist"}]
    result = match_event_windows(gt, [(250, "open", "fist")], intervals=intervals)
    assert (result["matched"], result["missed"], result["false_trigger"]) == (0, 1, 1)


def test_event_match_cannot_cross_right_hand_absence():
    intervals = [{"start_ms": 0, "end_ms": 100, "label": "open"},
                 {"start_ms": 100, "end_ms": 180, "label": "fist"},
                 {"start_ms": 180, "end_ms": 220, "label": "absent"},
                 {"start_ms": 220, "end_ms": 400, "label": "unknown"}]
    gt = [{"start_ms": 100, "end_ms": 100, "from": "open", "to": "fist"}]
    result = match_event_windows(gt, [(250, "open", "fist")], intervals=intervals)
    assert (result["matched"], result["missed"], result["false_trigger"]) == (0, 1, 1)
    assert not fully_labeled(intervals, 150, 250)


def test_group_metrics_recompute_from_counts_not_mean_of_clip_f1():
    matrix_a = {label: {pred: 0 for pred in ("open", "fist", "pinch", "unknown", "__no_output__")}
                for label in ("open", "fist", "pinch", "unknown")}
    matrix_b = json.loads(json.dumps(matrix_a))
    matrix_a["unknown"]["unknown"] = 1
    matrix_b["unknown"]["__no_output__"] = 9
    template = {"source_frames": 1, "labeled_target_frames": 1, "unlabeled_target_frames": 0,
                "gt_absent_target_frames": 0, "false_activation_on_absent_targets": 0,
                "prediction_count": 1, "fallback_count": 0, "no_output_count": 0,
                "base_late_count": 0, "prediction_late_count": 0, "labeled_output_count": 1,
                "labeled_correct_count": 1, "confusion": matrix_a, "fallback_reasons": {}}
    def clip(matrix, count):
        part = dict(template, source_frames=count, labeled_target_frames=count,
                    labeled_correct_count=matrix["unknown"]["unknown"],
                    confusion=matrix)
        forecast = {s: {m: {str(h): part for h in (50, 100, 150)} for m in
                        ("hold_channels", "hold_pose", "linear_pose_w2")}
                    for s in ("algorithm_only", "nominal", "jitter_drop", "compute_load")}
        return {"pose": {"visible_gt_points": count, "correct_points": matrix["unknown"]["unknown"],
                         "raw21_complete_frames": count, "gt_right_frames": count,
                         "by_sampling_purpose": {"uniform": {"frame_count": count, "visible_gt_points": count,
                                                              "correct_points": matrix["unknown"]["unknown"]}}},
                "gesture": {"confusion": matrix, "labeled_frames": count, "unlabeled_frames": 0,
                            "gt_absent_frames": 0, "false_activation_on_absent_frames": 0},
                "forecast": forecast}
    group = aggregate_group([clip(matrix_a, 1), clip(matrix_b, 9)])
    assert group["pose"]["full_gt_pck"] == 0.1
    assert group["gesture"]["per_class"]["unknown"]["f1"] == pytest.approx(2 / 11)
    assert group["forecast"]["nominal"]["linear_pose_w2"]["50"]["full_gt_gesture_accuracy"] == 0.1


def test_positive_pose_trajectory_runs_fixed_methods_and_four_scenarios(
        tmp_path, monkeypatch, synthetic_hand_pose):
    source = tmp_path / "pose.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 20, (640, 360))
    assert writer.isOpened()
    for i in range(12):
        writer.write(np.full((360, 640, 3), 200 - i, dtype=np.uint8))
    writer.release()
    dataset = tmp_path / "dataset"
    clip = import_clip(dataset, video=source, person_id="synthetic-p", session_id="synthetic-s",
                       clip_id="positive", scenario="synthetic-pose", role="development",
                       split_policy="independent_person", synthetic=True)
    media = load_timeline(clip)
    detected = encoded(synthetic_hand_pose("open")[1], width=640, height=360)
    raw = np.asarray(detected.landmarks_xyz, dtype=float)
    geometry = raw.copy()
    geometry[:, 1] *= 360 / 640
    manual_points = [{"x": x * 640, "y": y * 360, "visible": True} for x, y in detected.landmarks_2d]
    save_annotations(clip, {
        "schema_version": "camera-m3c-annotation-v1", "clip_id": "positive",
        "keypoint_frames": [{"frame_id": i, "presence": "right",
                             "sampling_purpose": "uniform" if i != 5 else "failure_targeted",
                             "points": manual_points} for i in (0, 5, 11)],
        "gesture_intervals": [{"start_ms": media[0]["pts_ms"], "end_ms": media[-1]["pts_ms"] + 50,
                               "label": "open"}], "transition_events": []})
    registry = read_json(dataset / "dataset.json")
    registry["clips"].append({"person_id": "heldout-p", "session_id": "heldout-s",
                              "clip_id": "not_opened", "scenario": "heldout",
                              "role": "heldout", "evidence_type": "real",
                              "video_path": str(tmp_path / "missing-heldout.avi")})
    (dataset / "dataset.json").write_text(json.dumps(registry), encoding="utf-8")

    def observations(entry, config, cfg, run_dir):
        return [{"video_id": entry["video_id"], "person_id": entry["person_id"],
                 "session_id": entry["session_id"], "role": entry["role"],
                 "frame_id": item["frame_id"], "source_time_ms": item["pts_ms"],
                 "timebase": "media_pts_ms", "timestamp_source": item["timestamp_source"],
                 "nominal_fps": 20.0, "image_width": 640, "image_height": 360,
                 "raw_landmarks_2d": detected.landmarks_2d, "raw_landmarks_xyz": raw.tolist(),
                 "geometry_landmarks": geometry.tolist(), "input_valid": True, "reason": "ok",
                 "segment_id": 0, "task_id": GEOMETRY_TASK, "read_ms": 1.0,
                 "detection_ms": 1.0, "baseline_mapping_ms": 1.0} for item in media]

    monkeypatch.setattr(camera_m3c, "extract_sequence", observations)
    path = camera_m3c.evaluate_dataset(dataset, load_config("configs/ai_m3a.yaml"),
                                       output_root=tmp_path / "reports")
    report = read_json(path / "report.json")
    assert report["evaluated_role"] == "development"
    assert len(report["clips"]) == 1
    assert report["clips"][0]["pose"]["full_gt_pck"] == 1
    assert set(report["clips"][0]["forecast"]) == {
        "algorithm_only", "nominal", "jitter_drop", "compute_load"}
    for scenario in report["clips"][0]["forecast"].values():
        assert set(scenario) == {"hold_channels", "hold_pose", "linear_pose_w2"}
        for method in scenario.values():
            assert set(method) == {"50", "100", "150"}
    assert report["groups"]["development/synthetic-p/synthetic-s"]["pose"]["full_gt_pck"] == 1
    assert report["clips"][0]["forecast"]["nominal"]["linear_pose_w2"]["100"]["prediction_count"] > 0
