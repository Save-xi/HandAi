"""M3-C 新会话导入、人工二维标签与真实标签评测。原始素材留在本机。"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import cv2
import numpy as np

from capture.timeline import resolve_media_timestamp_ms
from .camera_m3b import write_json
from .camera_replay import run_scenario
from .camera_sequence import extract_sequence, validate_sequence

GESTURES = ("open", "fist", "pinch", "unknown")
LABELS = (*GESTURES, "absent")
SAMPLING_PURPOSES = ("uniform", "failure_targeted", "dense_transition")
ROLES = ("development", "heldout")
HORIZONS = (50, 100, 150)
DEFAULT_SCENARIOS = (
    {"name": "nominal", "arrival_delay_cycle_ms": [0], "drop_every": 0,
     "detection_stall_ms": 0, "prediction_stall_ms": 0},
    {"name": "jitter_drop", "arrival_delay_cycle_ms": [0, 10, 20, 5], "drop_every": 17,
     "detection_stall_ms": 0, "prediction_stall_ms": 0},
    {"name": "compute_load", "arrival_delay_cycle_ms": [0], "drop_every": 0,
     "detection_stall_ms": 25, "prediction_stall_ms": 35},
)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"需要 JSON 对象：{path}")
    return value


def frame_timeline(video: Path) -> list[dict]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频：{video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or not 1 <= fps <= 240:
        cap.release()
        raise ValueError("视频名义 FPS 必须在 1..240")
    result: list[dict] = []
    previous = None
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            decision = resolve_media_timestamp_ms(
                len(result), raw_pts_ms=float(cap.get(cv2.CAP_PROP_POS_MSEC)),
                nominal_fps=fps, previous_timestamp_ms=previous,
            )
            result.append({"frame_id": len(result), "pts_ms": decision.timestamp_ms,
                           "timestamp_source": decision.source, "width": int(image.shape[1]),
                           "height": int(image.shape[0])})
            previous = decision.timestamp_ms
    finally:
        cap.release()
    if not result:
        raise ValueError("视频没有可解码帧")
    return result


def video_fps(video: Path) -> float:
    cap = cv2.VideoCapture(str(video))
    try:
        return float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()


def validate_registry(registry: dict) -> None:
    if registry.get("schema_version") != "camera-m3c-dataset-v1":
        raise ValueError("数据集版本无效")
    policy = registry.get("split_policy")
    if policy not in {"independent_person", "cross_session"}:
        raise ValueError("split_policy 必须为 independent_person 或 cross_session")
    clips = registry.get("clips")
    if not isinstance(clips, list):
        raise ValueError("clips 必须为列表")
    ids, paths, sessions, people = set(), set(), {}, {}
    for clip in clips:
        for key in ("person_id", "session_id", "clip_id", "scenario", "video_path"):
            if not isinstance(clip.get(key), str) or not clip[key].strip():
                raise ValueError(f"{key} 必须人工明确填写")
        if not all(character.isalnum() or character in "-_" for character in clip["clip_id"]) or clip["clip_id"] in {".", ".."}:
            raise ValueError("clip_id 只能包含字母、数字、-、_")
        if clip.get("role") not in ROLES:
            raise ValueError("role 只能为 development/heldout")
        if clip.get("evidence_type") not in {"real", "synthetic"}:
            raise ValueError("evidence_type 必须为 real/synthetic")
        if clip["clip_id"] in ids or clip["video_path"] in paths:
            raise ValueError("片段 ID 或源视频重复")
        ids.add(clip["clip_id"])
        paths.add(clip["video_path"])
        session = (clip["person_id"], clip["session_id"])
        if session in sessions and sessions[session] != clip["role"]:
            raise ValueError("同一人员会话不能跨用途")
        sessions[session] = clip["role"]
        if policy == "independent_person":
            person = clip["person_id"]
            if person in people and people[person] != clip["role"]:
                raise ValueError("独立人员切分不允许同人跨用途")
            people[person] = clip["role"]


def import_clip(root: Path, *, video: Path, person_id: str, session_id: str, clip_id: str,
                scenario: str, role: str, split_policy: str, synthetic: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    video = video.resolve(strict=True)
    index = root / "dataset.json"
    registry = read_json(index) if index.exists() else {
        "schema_version": "camera-m3c-dataset-v1", "split_policy": split_policy, "clips": []}
    if registry["split_policy"] != split_policy:
        raise ValueError("已有数据集的切分策略不同")
    entry = {"person_id": person_id, "session_id": session_id, "clip_id": clip_id,
             "scenario": scenario, "role": role, "video_path": str(video),
             "evidence_type": "synthetic" if synthetic else "real"}
    candidate = {**registry, "clips": registry["clips"] + [entry]}
    validate_registry(candidate)
    timeline = frame_timeline(video)
    clip_dir = root / clip_id
    if clip_dir.exists():
        raise ValueError("片段目录已存在")
    clip_dir.mkdir()
    with (clip_dir / "timeline.jsonl").open("w", encoding="utf-8") as handle:
        for row in timeline:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(clip_dir / "annotations.json", {
        "schema_version": "camera-m3c-annotation-v1", "clip_id": clip_id,
        "keypoint_frames": [], "gesture_intervals": [], "transition_events": [],
    })
    write_json(clip_dir / "manifest.json", {**entry, "nominal_fps": video_fps(video),
            "frame_count": len(timeline), "first_pts_ms": timeline[0]["pts_ms"],
            "last_pts_ms": timeline[-1]["pts_ms"], "timestamp_sources": dict(Counter(
                f["timestamp_source"] for f in timeline)), "width": timeline[0]["width"],
            "height": timeline[0]["height"], "timebase": "media_pts_ms"})
    write_json(index, candidate)
    return clip_dir


def load_timeline(clip_dir: Path) -> list[dict]:
    return [json.loads(line) for line in (clip_dir / "timeline.jsonl").read_text(encoding="utf-8").splitlines()]


def validate_annotations(annotation: dict, timeline: list[dict], clip_id: str) -> None:
    if annotation.get("schema_version") != "camera-m3c-annotation-v1" or annotation.get("clip_id") != clip_id:
        raise ValueError("标注版本或片段 ID 不匹配")
    if (not isinstance(annotation.get("keypoint_frames"), list)
            or not isinstance(annotation.get("gesture_intervals"), list)
            or not isinstance(annotation.get("transition_events"), list)):
        raise ValueError("标注列表无效")
    frame_ids = set()
    for row in annotation["keypoint_frames"]:
        index = row.get("frame_id")
        if type(index) is not int or not 0 <= index < len(timeline) or index in frame_ids:
            raise ValueError("关键点帧号越界或重复")
        frame_ids.add(index)
        if row.get("presence") not in {"right", "absent"}:
            raise ValueError("presence 必须为 right/absent；未标注应无此记录")
        if row.get("sampling_purpose") not in SAMPLING_PURPOSES:
            raise ValueError("sampling_purpose 须为 uniform/failure_targeted/dense_transition")
        if row["presence"] == "absent":
            if row.get("points") not in ([], None):
                raise ValueError("无右手帧不能含关键点")
            continue
        points = row.get("points")
        if not isinstance(points, list) or len(points) != 21:
            raise ValueError("右手必须含 21 个点")
        width, height = timeline[index]["width"], timeline[index]["height"]
        for point in points:
            if not isinstance(point, dict) or type(point.get("visible")) is not bool:
                raise ValueError("每点须明确可见或人工不可见；未完成帧不可保存为 GT")
            xy = (point.get("x"), point.get("y"))
            if point["visible"] and (any(type(v) not in (int, float) or not math.isfinite(v)
                                          for v in xy) or not (0 <= xy[0] < width and 0 <= xy[1] < height)):
                raise ValueError("可见关键点须在原图像素范围内")
            if not point["visible"] and any(v is not None for v in xy):
                raise ValueError("不可见点坐标须为 null")
    previous_end = -math.inf
    last_period = (timeline[-1]["pts_ms"] - timeline[-2]["pts_ms"] if len(timeline) > 1 else 1000.0)
    for interval in sorted(annotation["gesture_intervals"], key=lambda x: x.get("start_ms", -1)):
        start, end = interval.get("start_ms"), interval.get("end_ms")
        if interval.get("label") not in LABELS or any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end)):
            raise ValueError("手势区间标签或时间无效")
        if start < timeline[0]["pts_ms"] or end > timeline[-1]["pts_ms"] + last_period or end <= start or start < previous_end:
            raise ValueError("手势区间重叠、越界或非正长度")
        previous_end = end
    for event in annotation["transition_events"]:
        start, end = event.get("start_ms"), event.get("end_ms")
        if (event.get("from") not in GESTURES or event.get("to") not in GESTURES
                or event["from"] == event["to"]
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end))
                or start < timeline[0]["pts_ms"] or end > timeline[-1]["pts_ms"] + last_period
                or end < start):
            raise ValueError("转换事件须有同片段内的起止 PTS 及不同的 from/to 手势")
        if (not fully_labeled(annotation["gesture_intervals"], start, end)
                or labeled_segment_at(annotation["gesture_intervals"], (start + end) / 2) is None):
            raise ValueError("转换事件区间必须处于连续人工手势标注内，不能跨未标注孔洞")


def save_annotations(clip_dir: Path, annotation: dict) -> None:
    manifest = read_json(clip_dir / "manifest.json")
    validate_annotations(annotation, load_timeline(clip_dir), manifest["clip_id"])
    destination = clip_dir / "annotations.json"
    temporary = clip_dir / "annotations.json.tmp"
    write_json(temporary, annotation)
    temporary.replace(destination)


def label_at(intervals: list[dict], time_ms: float) -> str | None:
    for row in intervals:
        if row["start_ms"] <= time_ms < row["end_ms"]:
            return row["label"]
    return None


def match_events(gt: list[tuple], predicted: list[tuple], *, tolerance_ms: float = 250) -> dict:
    """同类型事件在容差内求最大一对一匹配，预测不能重复使用。"""
    options = [[pi for _, pi in sorted((abs(t - p), pi) for pi, (p, c, d) in enumerate(predicted)
                if (a, b) == (c, d) and abs(t - p) <= tolerance_ms)]
               for t, a, b in gt]
    owner: dict[int, int] = {}

    def augment(gi: int, seen: set[int]) -> bool:
        for pi in options[gi]:
            if pi in seen:
                continue
            seen.add(pi)
            if pi not in owner or augment(owner[pi], seen):
                owner[pi] = gi
                return True
        return False

    for gi in range(len(gt)):
        augment(gi, set())
    lags = [predicted[pi][0] - gt[gi][0] for pi, gi in sorted(owner.items())]
    return {"gt_events": len(gt), "predicted_events": len(predicted), "matched": len(lags),
            "missed": len(gt) - len(lags), "false_trigger": len(predicted) - len(lags),
            "lag_ms": lags, "median_lag_ms": float(np.median(lags)) if lags else None}


def labeled_segment_at(intervals: list[dict], at_ms: float) -> int | None:
    """按连续有手标签编号；未标注孔洞和右手离场均隔断事件。"""
    segment = -1
    previous_end = None
    previous_absent = False
    for interval in sorted(intervals, key=lambda row: row["start_ms"]):
        if interval["label"] == "absent":
            if interval["start_ms"] <= at_ms < interval["end_ms"]:
                return None
            previous_end = interval["end_ms"]
            previous_absent = True
            continue
        if previous_end is None or previous_absent or interval["start_ms"] > previous_end + 1e-6:
            segment += 1
        previous_absent = False
        if interval["start_ms"] <= at_ms < interval["end_ms"]:
            return segment
        previous_end = interval["end_ms"]
    return None


def match_event_windows(gt: list[dict], predicted: list[tuple], *, tolerance_ms: float = 250,
                        intervals: list[dict] | None = None) -> dict:
    """人工起止区间内为零滞后；窗外按最近边界记符号，仍一对一最大匹配。"""
    options = []
    for gi, event in enumerate(gt):
        candidates = []
        gt_segment = (labeled_segment_at(intervals, (event["start_ms"] + event["end_ms"]) / 2)
                      if intervals is not None else None)
        for pi, (at, before, after) in enumerate(predicted):
            if (before, after) != (event["from"], event["to"]):
                continue
            if intervals is not None and (
                    gt_segment is None or not fully_labeled(intervals, event["start_ms"], event["end_ms"])
                    or labeled_segment_at(intervals, at) != gt_segment):
                continue
            lag = at - event["end_ms"] if at > event["end_ms"] else (
                at - event["start_ms"] if at < event["start_ms"] else 0.0)
            if abs(lag) <= tolerance_ms:
                candidates.append((abs(lag), pi, lag))
        options.append(sorted(candidates))
    owner: dict[int, tuple[int, float]] = {}

    def augment(gi: int, seen: set[int]) -> bool:
        for _, pi, lag in options[gi]:
            if pi in seen:
                continue
            seen.add(pi)
            if pi not in owner or augment(owner[pi][0], seen):
                owner[pi] = (gi, lag)
                return True
        return False

    for gi in range(len(gt)):
        augment(gi, set())
    lags = [lag for _, lag in owner.values()]
    return {"gt_events": len(gt), "predicted_events": len(predicted), "matched": len(lags),
            "missed": len(gt) - len(lags), "false_trigger": len(predicted) - len(lags),
            "lag_ms": lags, "median_lag_ms": float(np.median(lags)) if lags else None,
            "lag_definition": "窗口内0；提前为负，滞后为正；窗外按最近边界"}


def ground_truth_events(annotation: dict) -> list[dict]:
    if annotation["transition_events"]:
        return annotation["transition_events"]
    return [{"start_ms": at, "end_ms": at, "from": before, "to": after}
            for at, before, after in interval_events(annotation["gesture_intervals"])]


def interval_events(intervals: list[dict]) -> list[tuple]:
    ordered = sorted(intervals, key=lambda row: row["start_ms"])
    return [(right["start_ms"], left["label"], right["label"]) for left, right in zip(ordered, ordered[1:])
            if abs(left["end_ms"] - right["start_ms"]) < 1e-6 and left["label"] != right["label"]
            and left["label"] in GESTURES and right["label"] in GESTURES]


def fully_labeled(intervals: list[dict], start_ms: float, end_ms: float) -> bool:
    cursor = start_ms
    for interval in sorted(intervals, key=lambda row: row["start_ms"]):
        if interval["end_ms"] <= cursor:
            continue
        if interval["start_ms"] > cursor + 1e-6:
            return False
        if interval["label"] == "absent":
            return False
        cursor = max(cursor, interval["end_ms"])
        if cursor >= end_ms - 1e-6:
            return True
    return False


def predicted_events(rows: list[dict], intervals: list[dict]) -> list[tuple]:
    ordered = sorted(rows, key=lambda row: row["target_time_ms"])
    events, previous = [], None
    for row in ordered:
        label = label_at(intervals, row["target_time_ms"])
        if label is None:
            previous = None
            continue
        if (previous and row["frame_id"] == previous["frame_id"] + 1
                and fully_labeled(intervals, previous["target_time_ms"], row["target_time_ms"])
                and row["gesture"] in GESTURES):
            if previous["gesture"] in GESTURES and previous["gesture"] != row["gesture"]:
                events.append((row["target_time_ms"], previous["gesture"], row["gesture"]))
        previous = row
    return events


def valid_raw_xy(points: object) -> bool:
    return isinstance(points, list) and len(points) == 21 and all(
        isinstance(point, list) and len(point) == 2 and all(
            type(value) in (int, float) and math.isfinite(value) for value in point)
        for point in points)


def score_pose(annotation: dict, observations: list[dict], timeline: list[dict], threshold_px: float) -> dict:
    if threshold_px <= 0:
        raise ValueError("PCK 阈值须为正数")
    count, hit, output_points, pixel_errors = 0, 0, 0, []
    gt_right, detected_right, missed_right, gt_absent, false_right = 0, 0, 0, 0, 0
    examples = []
    for row in annotation["keypoint_frames"]:
        index = row["frame_id"]
        observation = observations[index]
        prediction = observation.get("raw_landmarks_2d", [])
        prediction_valid = valid_raw_xy(prediction)
        if row["presence"] == "absent":
            gt_absent += 1
            false_right += int(bool(prediction_valid))
            continue
        gt_right += 1
        detected_right += int(bool(prediction_valid))
        missed_right += int(not prediction_valid)
        width, height = timeline[index]["width"], timeline[index]["height"]
        frame_errors = []
        for joint, point in enumerate(row["points"]):
            if not point["visible"]:
                continue
            count += 1
            if prediction_valid:
                xy = prediction[joint]
                if len(xy) != 2 or not all(type(v) in (int, float) and math.isfinite(v) for v in xy):
                    continue
                distance = math.hypot(xy[0] * width - point["x"], xy[1] * height - point["y"])
                output_points += 1
                hit += int(distance <= threshold_px)
                pixel_errors.append(distance)
                frame_errors.append(distance)
        if not prediction_valid or frame_errors and max(frame_errors) > threshold_px:
            examples.append({"frame_id": index, "pts_ms": timeline[index]["pts_ms"],
                             "reason": observation.get("reason"), "control_valid": observation.get("input_valid"),
                             "max_visible_error_px": max(frame_errors) if frame_errors else None,
                             "candidate_cause": "待人工复核：漏检/遮挡/视角/几何边界"})
    by_purpose = {}
    for purpose in SAMPLING_PURPOSES:
        selected = [row for row in annotation["keypoint_frames"] if row["sampling_purpose"] == purpose]
        if selected:
            subset_count = subset_hit = 0
            for item in selected:
                if item["presence"] != "right":
                    continue
                index = item["frame_id"]
                raw = observations[index].get("raw_landmarks_2d", [])
                raw_valid = valid_raw_xy(raw)
                width, height = timeline[index]["width"], timeline[index]["height"]
                for joint, point in enumerate(item["points"]):
                    if not point["visible"]:
                        continue
                    subset_count += 1
                    if raw_valid:
                        subset_hit += int(math.hypot(raw[joint][0] * width - point["x"],
                                                     raw[joint][1] * height - point["y"]) <= threshold_px)
            by_purpose[purpose] = {"frame_count": len(selected), "visible_gt_points": subset_count,
                                   "correct_points": subset_hit,
                                   "full_gt_pck": subset_hit / subset_count if subset_count else None}
    return {"threshold_px": threshold_px, "visible_gt_points": count, "correct_points": hit,
            "full_gt_pck": hit / count if count else None, "prediction_visible_points": output_points,
            "conditional_pck": hit / output_points if output_points else None,
            "conditional_mean_pixel_error": float(np.mean(pixel_errors)) if pixel_errors else None,
            "visible_point_coverage": output_points / count if count else None,
            "gt_right_frames": gt_right, "detected_right_frames": detected_right,
            "missed_right_frames": missed_right, "gt_absent_frames": gt_absent,
            "false_right_frames": false_right, "unlabeled_frames": len(timeline) - len(annotation["keypoint_frames"]),
            "detection_is_raw_right_hand": True,
            "raw21_complete_frames": sum(
                valid_raw_xy(observations[row["frame_id"]].get("raw_landmarks_2d"))
                for row in annotation["keypoint_frames"] if row["presence"] == "right"),
            "by_sampling_purpose": by_purpose,
            "failure_examples": examples[:30]}


def score_gesture(intervals: list[dict], observations: list[dict], timeline: list[dict],
                  transition_events: list[dict] | None = None) -> dict:
    confusion = {label: Counter() for label in GESTURES}
    unlabeled = absent = false_activation = 0
    examples = []
    for frame, observation in zip(timeline, observations):
        truth = label_at(intervals, frame["pts_ms"])
        if truth is None:
            unlabeled += 1
            continue
        if truth == "absent":
            absent += 1
            false_activation += int(bool(observation.get("input_valid")))
            continue
        predicted = observation.get("gesture_stable") if observation.get("input_valid") else "__no_output__"
        if predicted not in GESTURES and predicted != "__no_output__":
            predicted = "__no_output__"
        confusion[truth][predicted] += 1
        if truth != predicted:
            examples.append({"frame_id": frame["frame_id"], "pts_ms": frame["pts_ms"],
                             "gt": truth, "predicted": predicted, "quality_reason": observation.get("reason"),
                             "candidate_cause": "待人工复核：检测/遮挡/规则边界/转换"})
    matrix = {label: {pred: confusion[label][pred] for pred in (*GESTURES, "__no_output__")} for label in GESTURES}
    per_class = {}
    for label in GESTURES:
        tp = confusion[label][label]
        actual = sum(confusion[label].values())
        predicted = sum(confusion[truth][label] for truth in GESTURES)
        precision = tp / predicted if predicted else None
        recall = tp / actual if actual else None
        per_class[label] = {"support": actual, "precision": precision, "recall": recall,
                            "f1": 2 * tp / (actual + predicted) if actual + predicted else None}
    source_rows = [{"frame_id": frame["frame_id"], "target_time_ms": frame["pts_ms"],
                    "gesture": observation.get("gesture_stable") if observation.get("input_valid") else None}
                   for frame, observation in zip(timeline, observations)]
    truth_events = (transition_events if transition_events is not None else
                    ground_truth_events({"gesture_intervals": intervals, "transition_events": []}))
    inside = [event for event in truth_events if event["end_ms"] >= timeline[0]["pts_ms"]
              and event["start_ms"] <= timeline[-1]["pts_ms"]]
    return {"labeled_frames": len(timeline) - unlabeled - absent, "unlabeled_frames": unlabeled,
            "gt_absent_frames": absent, "false_activation_on_absent_frames": false_activation,
            "confusion": matrix, "per_class": per_class,
            "transitions": match_event_windows(inside, predicted_events(source_rows, intervals),
                                                intervals=intervals),
            "transition_events_outside_target_range": len(truth_events) - len(inside),
            "failure_examples": examples[:30]}


def score_forecast(rows: list[dict], intervals: list[dict], total_frames: int,
                   transition_events: list[dict] | None = None) -> dict:
    result = {}
    for horizon in HORIZONS:
        selected = [row for row in rows if row["horizon_ms"] == horizon]
        if not selected:
            continue
        truth_events = (transition_events if transition_events is not None else
                        ground_truth_events({"gesture_intervals": intervals, "transition_events": []}))
        target_min = min(row["target_time_ms"] for row in selected)
        target_max = max(row["target_time_ms"] for row in selected)
        eligible_events = [event for event in truth_events if event["end_ms"] >= target_min
                           and event["start_ms"] <= target_max]
        labeled = [(row, label_at(intervals, row["target_time_ms"])) for row in selected]
        absent = [(row, label) for row, label in labeled if label == "absent"]
        labeled = [(row, label) for row, label in labeled if label in GESTURES]
        correct = sum(row["gesture"] == label for row, label in labeled if row["gesture"] is not None)
        available = sum(row["gesture"] is not None for row, _ in labeled)
        confusion = {label: Counter() for label in GESTURES}
        for row, truth in labeled:
            confusion[truth][row["gesture"] if row["gesture"] in GESTURES else "__no_output__"] += 1
        per_class = {}
        for label in GESTURES:
            tp = confusion[label][label]
            support = sum(confusion[label].values())
            predicted_count = sum(confusion[truth][label] for truth in GESTURES)
            precision = tp / predicted_count if predicted_count else None
            recall = tp / support if support else None
            per_class[label] = {"support": support, "precision": precision, "recall": recall,
                                "f1": 2 * tp / (support + predicted_count) if support + predicted_count else None}
        base_late = sum(row["timing"]["baseline_ready_ms"] is not None and
                        row["timing"]["baseline_ready_ms"] > row["target_time_ms"] for row in selected)
        prediction_late = sum(row["timing"]["baseline_ready_ms"] is not None and
                              row["timing"]["baseline_ready_ms"] <= row["target_time_ms"] and
                              row["ready_time_ms"] is not None and row["ready_time_ms"] > row["target_time_ms"]
                              for row in selected)
        result[str(horizon)] = {
            "source_frames": total_frames, "labeled_target_frames": len(labeled),
            "unlabeled_target_frames": len(selected) - len(labeled) - len(absent),
            "gt_absent_target_frames": len(absent),
            "false_activation_on_absent_targets": sum(row["gesture"] in GESTURES for row, _ in absent),
            "prediction_count": sum(row["prediction_on_time"] for row in selected),
            "fallback_count": sum(row["used_fallback"] for row in selected),
            "no_output_count": sum(not row["valid"] for row in selected),
            "fallback_reasons": dict(Counter(row["fallback_reason"] for row in selected
                                             if row["fallback_reason"] is not None)),
            "base_late_count": base_late, "prediction_late_count": prediction_late,
            "labeled_output_count": available, "labeled_correct_count": correct,
            "full_gt_gesture_accuracy": correct / len(labeled) if labeled else None,
            "conditional_gesture_accuracy": correct / available if available else None,
            "confusion": {label: {pred: confusion[label][pred] for pred in (*GESTURES, "__no_output__")}
                          for label in GESTURES},
            "per_class": per_class,
            "transitions": match_event_windows(eligible_events, predicted_events(selected, intervals),
                                                intervals=intervals),
            "transition_events_outside_target_range": len(truth_events) - len(eligible_events),
            "failure_examples": [
                {"frame_id": row["frame_id"], "target_time_ms": row["target_time_ms"],
                 "gt": label, "predicted": row["gesture"], "fallback_reason": row["fallback_reason"],
                 "base_late": row["timing"]["baseline_ready_ms"] is not None
                 and row["timing"]["baseline_ready_ms"] > row["target_time_ms"],
                 "prediction_late": row["ready_time_ms"] is not None and row["ready_time_ms"] > row["target_time_ms"],
                 "candidate_cause": "待人工复核：运动/检测/规则/基础流或预测逾期"}
                for row, label in labeled if row["gesture"] != label][:30]}
    return result


def summarize_confusion(matrix: dict) -> dict:
    per_class = {}
    for label in GESTURES:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[truth][label] for truth in GESTURES)
        per_class[label] = {
            "support": support, "precision": tp / predicted if predicted else None,
            "recall": tp / support if support else None,
            "f1": 2 * tp / (support + predicted) if support + predicted else None,
        }
    return per_class


def sum_confusion(matrices: list[dict]) -> dict:
    return {truth: {pred: sum(matrix[truth][pred] for matrix in matrices)
                    for pred in (*GESTURES, "__no_output__")} for truth in GESTURES}


def aggregate_group(rows: list[dict]) -> dict:
    pose = {"visible_gt_points": sum(row["pose"]["visible_gt_points"] for row in rows),
            "correct_points": sum(row["pose"]["correct_points"] for row in rows),
            "raw21_complete_frames": sum(row["pose"]["raw21_complete_frames"] for row in rows),
            "gt_right_frames": sum(row["pose"]["gt_right_frames"] for row in rows)}
    pose["full_gt_pck"] = (pose["correct_points"] / pose["visible_gt_points"]
                            if pose["visible_gt_points"] else None)
    pose["raw21_complete_rate_on_labeled_right"] = (pose["raw21_complete_frames"] / pose["gt_right_frames"]
                                                     if pose["gt_right_frames"] else None)
    by_purpose = {}
    for purpose in SAMPLING_PURPOSES:
        subsets = [row["pose"]["by_sampling_purpose"].get(purpose, {}) for row in rows]
        frames = sum(part.get("frame_count", 0) for part in subsets)
        gt = sum(part.get("visible_gt_points", 0) for part in subsets)
        hits = sum(part.get("correct_points", 0) for part in subsets)
        by_purpose[purpose] = {"frame_count": frames, "visible_gt_points": gt, "correct_points": hits,
                               "full_gt_pck": hits / gt if gt else None}
    pose["by_sampling_purpose"] = by_purpose
    gesture_matrix = sum_confusion([row["gesture"]["confusion"] for row in rows])
    gesture = {"confusion": gesture_matrix, "per_class": summarize_confusion(gesture_matrix),
               "labeled_frames": sum(row["gesture"]["labeled_frames"] for row in rows),
               "unlabeled_frames": sum(row["gesture"]["unlabeled_frames"] for row in rows),
               "gt_absent_frames": sum(row["gesture"]["gt_absent_frames"] for row in rows),
               "false_activation_on_absent_frames": sum(row["gesture"]["false_activation_on_absent_frames"]
                                                         for row in rows)}
    forecast = {}
    for scenario in rows[0]["forecast"]:
        forecast[scenario] = {}
        for method in rows[0]["forecast"][scenario]:
            forecast[scenario][method] = {}
            for horizon in HORIZONS:
                parts = [row["forecast"][scenario][method][str(horizon)] for row in rows]
                matrix = sum_confusion([part["confusion"] for part in parts])
                totals = {key: sum(part[key] for part in parts) for key in (
                    "source_frames", "labeled_target_frames", "unlabeled_target_frames",
                    "gt_absent_target_frames", "false_activation_on_absent_targets",
                    "prediction_count", "fallback_count", "no_output_count",
                    "base_late_count", "prediction_late_count", "labeled_output_count",
                    "labeled_correct_count")}
                totals["full_gt_gesture_accuracy"] = (totals["labeled_correct_count"] / totals["labeled_target_frames"]
                                                        if totals["labeled_target_frames"] else None)
                totals["conditional_gesture_accuracy"] = (totals["labeled_correct_count"] / totals["labeled_output_count"]
                                                            if totals["labeled_output_count"] else None)
                totals["prediction_coverage_all_sources"] = totals["prediction_count"] / totals["source_frames"]
                totals["output_coverage_all_sources"] = 1 - totals["no_output_count"] / totals["source_frames"]
                totals["confusion"] = matrix
                totals["per_class"] = summarize_confusion(matrix)
                totals["fallback_reasons"] = dict(sum((Counter(part["fallback_reasons"]) for part in parts), Counter()))
                forecast[scenario][method][str(horizon)] = totals
    return {"clip_count": len(rows), "pose": pose, "gesture": gesture, "forecast": forecast,
            "uncertainty_note": "会话聚合计数；不把帧当独立人员，也不生成置信区间"}


def evaluate_dataset(root: Path, cfg: dict, *, threshold_px: float = 20,
                     output_root: Path | None = None, scenarios: tuple[dict, ...] = DEFAULT_SCENARIOS,
                     role: str = "development") -> Path:
    from .camera_sequence import sequence_summary
    from svh.mapping_contract import mapping_contract_payload

    registry = read_json(root / "dataset.json")
    validate_registry(registry)
    if role not in ROLES:
        raise ValueError("评测用途必须为 development/heldout")
    if not any(clip["role"] == role for clip in registry["clips"]):
        raise ValueError(f"数据集没有 {role} 片段，未运行评测")
    if cfg.get("geometry_mode") != "image_width_xyz_v1" or cfg.get("control_open_release_mode") != "continuous_v1":
        raise ValueError("M3-C 只接受 M3-A 几何与连续释放配置")
    output_root = output_root or root / "reports"
    run_dir = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": "camera-m3c-real-label-report-v1", "split_policy": registry["split_policy"],
              "evaluated_role": role,
              "mapping": mapping_contract_payload(cfg), "selected_baseline": "linear_pose_w2",
              "comparison_baselines": ["hold_channels", "hold_pose"],
              "pose_truth": "人工二维原图像素；无公制三维或 9 通道真值",
              "cost_timing": "实测成本驱动离散事件重放；不是墙钟并发/设备实测",
              "clips": [], "groups": {}, "people": {}, "independent_effect_claim": False}
    write_json(run_dir / "evaluation_config.json", {
        "runtime_config": cfg, "pck_threshold_px": threshold_px, "event_tolerance_ms": 250,
        "prediction_fps": 30.0, "methods": ["hold_channels", "hold_pose", "linear_pose_w2"],
        "scenarios": [{"name": "algorithm_only", "cost_mode": "zero"}, *scenarios],
        "evaluated_role": role, "split_policy": registry["split_policy"]})
    for clip in registry["clips"]:
        if clip["role"] != role:
            continue
        clip_dir = root / clip["clip_id"]
        manifest = read_json(clip_dir / "manifest.json")
        annotation = read_json(clip_dir / "annotations.json")
        timeline = load_timeline(clip_dir)
        validate_annotations(annotation, timeline, clip["clip_id"])
        write_json(run_dir / f"{clip['clip_id']}.annotations.json", annotation)
        write_json(run_dir / f"{clip['clip_id']}.manifest.json", manifest)
        with (run_dir / f"{clip['clip_id']}.timeline.jsonl").open("w", encoding="utf-8") as handle:
            for item in timeline:
                handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
        if any(manifest.get(key) != clip[key] for key in ("person_id", "session_id", "clip_id", "role", "video_path",
                                                         "evidence_type")):
            raise ValueError("数据集清单与片段来源不一致")
        if Path(clip["video_path"]).resolve() != Path(manifest["video_path"]).resolve():
            raise ValueError("源视频路径不一致")
        entry = {"video_id": clip["clip_id"], "person_id": clip["person_id"],
                 "session_id": clip["session_id"], "role": clip["role"],
                 "file": clip["video_path"]}
        settings = {"video_root": "", "max_frames_per_video": len(timeline)}
        frames = extract_sequence(entry, settings, cfg, run_dir)
        validate_sequence(frames, cfg)
        if len(frames) != len(timeline) or any(
                row["frame_id"] != source["frame_id"] or
                abs(row["source_time_ms"] - source["pts_ms"]) > 1e-6 or
                (row["image_width"], row["image_height"]) != (source["width"], source["height"])
                for row, source in zip(frames, timeline)):
            raise ValueError("检测帧与导入时的原尺寸/PTS 不一致")
        pose = score_pose(annotation, frames, timeline, threshold_px)
        # 手势基础流必须由同一个 HandPipeline 在观测点逐帧推进。
        from pipeline import HandPipeline
        from prediction.camera_baselines import observe_frame
        pipeline, previous, actual = HandPipeline(cfg), None, []
        for row in frames:
            actual.append(observe_frame(pipeline, row, previous))
            previous = row
        gesture_rows = [{**row, "gesture_stable": payload["gesture_stable"]} for row, payload in zip(frames, actual)]
        gestures = score_gesture(annotation["gesture_intervals"], gesture_rows, timeline,
                                 ground_truth_events(annotation))
        replay = {}
        for scenario in ({**scenarios[0], "name": "algorithm_only"}, *scenarios):
            algorithm_only = scenario["name"] == "algorithm_only"
            records = run_scenario(frames, cfg, methods=[("hold_channels", 2), ("hold_pose", 2),
                                   ("linear_pose", 2)], fps=30.0, scenario=scenario,
                                   algorithm_only=algorithm_only)
            replay[scenario["name"]] = {name: score_forecast(rows, annotation["gesture_intervals"], len(timeline),
                                                               ground_truth_events(annotation))
                                         for name, rows in records.items()}
            for name, rows in records.items():
                with (run_dir / f"{clip['clip_id']}.{scenario['name']}.{name}.jsonl").open("w", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        report["clips"].append({**clip, "source": sequence_summary(frames, cfg), "pose": pose,
                                "gesture": gestures, "forecast": replay})
    groups = defaultdict(list)
    people = defaultdict(list)
    for row in report["clips"]:
        groups[(row["role"], row["person_id"], row["session_id"])].append(row)
        people[(row["role"], row["person_id"])].append(row)
    report["groups"] = {"/".join(key): aggregate_group(rows) for key, rows in groups.items()}
    report["people"] = {"/".join(key): aggregate_group(rows) for key, rows in people.items()}
    heldout_groups = {c["person_id"] if registry["split_policy"] == "independent_person"
                      else (c["person_id"], c["session_id"]) for c in registry["clips"] if c["role"] == "heldout"}
    report["independent_evidence_eligible"] = bool(len(heldout_groups) >= 2 and all(
        c["pose"]["visible_gt_points"] and c["gesture"]["labeled_frames"]
        for c in report["clips"] if c["role"] == "heldout") and all(
            c["evidence_type"] == "real" for c in report["clips"] if c["role"] == "heldout"))
    if role != "heldout":
        report["independent_evidence_eligible"] = False
    report["independent_effect_claim"] = False
    report["claim_note"] = ("独立组和标签数量具备评审条件；须人工复核来源与标注后解释效果" if report["independent_evidence_eligible"]
                            else "独立人员/会话或人工标签不足；工具运行不等于 M3-C 真实效果验收")
    write_json(run_dir / "report.json", report)
    return run_dir
